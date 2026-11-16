"""Editable PowerPoint copy of the deck, with the same click steps.

    uv run --no-project --with python-pptx python slides/build_pptx.py     # -> slides/copd_social_needs_slides.pptx
    uv run --no-project --with python-pptx python slides/build_pptx.py --main-only --google --out FILE.pptx

--main-only leaves out the backup section (the 12 slides that are presented). --google tunes the text
boxes for Google Slides, whose import wraps every box at its edge, may drop kerning, turns line spacing
into a multiple of 1.2 times the font size and gives a bare line break 18 pt Calibri. The Google
Slides copy is that file uploaded to Google Drive and converted (see AGENTS.md section 7).

The slide source stays slides/deck/project/. This script lays the slides out in headless Chrome
(the page export_static_pdf.py builds), reads back where every heading, paragraph, box and image
ended up, and rebuilds each slide from native PowerPoint objects at those positions:

    text      -> text boxes in the deck's faces, Inter and Inter Tight (the line breaks of the web
                 version are kept). Google Slides has both faces; for PowerPoint or Keynote install
                 the files in slides/fonts/ first, or the text falls back to another sans-serif.
    boxes     -> rectangles and rounded rectangles
    figures   -> the same transparent PNG layers, stacked
    fragments -> "fade in on click" animations, one click per data-fragment-index
    <aside>   -> speaker notes

python-pptx is not in requirements.txt on purpose: the analysis does not need it. Needs Chrome,
Chromium or Edge. Run export_static_pdf.py --check first if a figure was regenerated.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

import export_static_pdf as deck

OUT = deck.SLIDES / "copd_social_needs_slides.pptx"
PX = 6350                      # EMU per slide pixel: 1920 px across a 13.333 inch slide
# ascent and descent in em of the deck's faces (OS/2 typo metrics of slides/fonts/: 1984 and 494 of
# 2048 units for both), which place a line's baseline in its line box
METRICS = {"Inter": (0.96875, 0.24121), "Inter Tight": (0.96875, 0.24121)}

# Runs in the local page. Collects, per slide, the painted boxes, images and text blocks in paint
# order, each with its position, its click step and (for text) the lines as the browser broke them.
DUMP_SCRIPT = r"""
const measure = document.createElement('canvas').getContext('2d');
function lineWidth(text, cs, family) {
  measure.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${family}`;
  const spacing = parseFloat(cs.letterSpacing) || 0;
  return measure.measureText(text).width + spacing * text.length;
}
function linesOf(el) {
  const lines = [];
  let current = null, top = null;
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const cs = getComputedStyle(node.parentElement);
    const style = {bold: Number(cs.fontWeight) >= 600, italic: cs.fontStyle === 'italic', color: cs.color};
    const range = document.createRange();
    const place = (from, to) => {
      range.setStart(node, from);
      range.setEnd(node, to);
      return range.getClientRects()[0];
    };
    const add = (text, rect) => {
      if (top === null || Math.abs(rect.top - top) > rect.height * 0.5) { current = []; lines.push(current); top = rect.top; }
      current.push({...style, text});
    };
    const re = /\S+|\s+/g;
    for (let m = re.exec(node.textContent); m; m = re.exec(node.textContent)) {
      if (/^\s+$/.test(m[0])) { if (current && current.length) current.push({...style, text: ' '}); continue; }
      range.setStart(node, m.index);
      range.setEnd(node, m.index + m[0].length);
      const rects = range.getClientRects();
      if (!rects.length) continue;
      if (rects.length === 1) { add(m[0], rects[0]); continue; }
      // the browser broke this word (after a hyphen): give each character to the line it sits on
      for (let k = 0; k < m[0].length; k++) {
        const rect = place(m.index + k, m.index + k + 1);
        if (rect) add(m[0][k], rect);
      }
    }
  }
  return lines.map(runs => {
    while (runs.length && runs[runs.length - 1].text === ' ') runs.pop();
    const merged = [];
    for (const run of runs) {
      const last = merged[merged.length - 1];
      if (last && last.bold === run.bold && last.italic === run.italic && last.color === run.color) last.text += run.text;
      else merged.push({...run});
    }
    return merged;
  });
}
// measure only once every face is loaded, or the lines would break as in a fallback face
Promise.all(Array.from(document.fonts, face => face.load())).then(() => document.fonts.ready).then(() => {
const slides = [];
let fragmentCount = 0;
document.querySelectorAll('body > section').forEach((section, index) => {
  const origin = section.getBoundingClientRect();
  const items = [];
  const aside = section.querySelector('aside');
  const walk = (el, step, fragment) => {
    if (el.tagName === 'ASIDE') return;
    const cs = getComputedStyle(el);
    if (cs.display === 'none') return;
    if (el.classList.contains('fragment')) { step = Number(el.dataset.fragmentIndex); fragment = ++fragmentCount; }
    const r = el.getBoundingClientRect();
    const box = {x: r.left - origin.left, y: r.top - origin.top, w: r.width, h: r.height, step, fragment};
    const edges = {};
    for (const side of ['Top', 'Bottom']) {
      const width = parseFloat(cs[`border${side}Width`]);
      if (width > 0 && cs[`border${side}Style`] !== 'none') edges[side.toLowerCase()] = {width, color: cs[`border${side}Color`]};
    }
    if (cs.backgroundColor !== 'rgba(0, 0, 0, 0)' || Object.keys(edges).length) {
      items.push({kind: 'box', ...box, fill: cs.backgroundColor, radius: parseFloat(cs.borderTopLeftRadius) || 0, edges});
    }
    if (el.tagName === 'IMG') { items.push({kind: 'image', ...box, src: el.currentSrc || el.src, alt: el.alt}); return; }
    const isNumber = el.classList.contains('pageno');
    if (['H1', 'H2', 'H3', 'P'].includes(el.tagName) || isNumber) {
      const pad = side => parseFloat(cs[`padding${side}`]) || 0;
      const lines = isNumber ? [[{text: String(index + 1), bold: false, italic: false, color: cs.color}]] : linesOf(el);
      if (!lines.length) return;
      const family = cs.fontFamily.split(',')[0].replace(/["']/g, '').trim();
      const shown = Math.max(...lines.map(runs => lineWidth(runs.map(run => run.text).join(''), cs, cs.fontFamily)));
      items.push({kind: 'text', tag: el.tagName, cls: el.className,
                  x: box.x + pad('Left'), y: box.y + pad('Top'), w: box.w - pad('Left') - pad('Right'),
                  h: box.h - pad('Top') - pad('Bottom'), step, fragment, lines, family,
                  size: parseFloat(cs.fontSize), weight: Number(cs.fontWeight), color: cs.color,
                  align: isNumber ? 'right' : cs.textAlign, lineHeight: parseFloat(cs.lineHeight),
                  spacing: parseFloat(cs.letterSpacing) || 0, shown});
      return;
    }
    for (const child of el.children) walk(child, step, fragment);
  };
  for (const child of section.children) walk(child, 0, 0);
  slides.push({id: section.id, background: getComputedStyle(section).backgroundColor, items,
               notes: aside ? aside.textContent : ''});
});
const holder = document.createElement('pre');
holder.id = 'layout-dump';
holder.textContent = JSON.stringify(slides);
document.body.appendChild(holder);
});
"""

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def measured_layout() -> list[dict]:
    index, asset_map = deck.load_index(), deck.load_asset_map()
    stale = deck.stale_assets(asset_map)
    if stale:
        raise SystemExit("Images changed since upload, so the web deck and this file would differ:\n  "
                         + "\n  ".join(stale))
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "deck.html"
        page.write_text(deck.page_html(index["order"], index, asset_map, extra_script=DUMP_SCRIPT), encoding="utf-8")
        dom = subprocess.run([deck.browser(), "--headless=new", "--disable-gpu", "--virtual-time-budget=8000",
                              "--dump-dom", page.as_uri()], check=True, capture_output=True, text=True).stdout
    found = re.search(r'<pre id="layout-dump">(.*?)</pre>', dom, flags=re.S)
    if not found:
        raise SystemExit("The page did not report its layout.")
    text = found.group(1).replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")
    return json.loads(text)


def colour(css: str) -> RGBColor:
    r, g, b = (int(float(v)) for v in re.findall(r"[\d.]+", css)[:3])
    return RGBColor(r, g, b)


def emu(pixels: float) -> Emu:
    return Emu(round(pixels * PX))


def plain(shape, fill: str) -> None:
    """Flat fill, no outline, and none of the theme's shape style (which would add a shadow)."""
    shape.fill.solid()
    shape.fill.fore_color.rgb = colour(fill)
    shape.line.fill.background()
    style = shape._element.find(qn("p:style"))
    if style is not None:
        shape._element.remove(style)


def add_box(shapes, item: dict) -> list:
    made = []
    if item["fill"] != "rgba(0, 0, 0, 0)":
        rounded = item["radius"] > 0
        shape = shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                 emu(item["x"]), emu(item["y"]), emu(item["w"]), emu(item["h"]))
        if rounded:
            shape.adjustments[0] = min(0.5, item["radius"] / min(item["w"], item["h"]))
        plain(shape, item["fill"])
        made.append(shape)
    for side, edge in item["edges"].items():          # a hairline above or below a block
        y = item["y"] if side == "top" else item["y"] + item["h"] - edge["width"]
        line = shapes.add_shape(MSO_SHAPE.RECTANGLE, emu(item["x"]), emu(y), emu(item["w"]), emu(edge["width"]))
        plain(line, edge["color"])
        made.append(line)
    return made


def add_image(shapes, item: dict):
    path = unquote(urlparse(item["src"]).path)
    picture = shapes.add_picture(path, emu(item["x"]), emu(item["y"]), emu(item["w"]), emu(item["h"]))
    picture._element.nvPicPr.cNvPr.set("descr", item["alt"])
    return picture


def add_text(shapes, item: dict, google: bool = False):
    size = round(item["size"])                     # in slide pixels; the run size is half of it in points
    family = item["family"]
    line = item["lineHeight"]
    # lines are not re-wrapped, so the box only has to be as wide as the longest one (plus a little
    # slack for a renderer that sets the face a little wider); it grows away from the side the text is
    # aligned to. Google Slides wraps at the edge and may drop kerning, so it gets half an em more.
    x, y, width = item["x"], item["y"], item["w"]
    extra = item["shown"] * 1.03 + (0.5 * size if google else 0) - width
    if extra > 0:
        width += extra
        x -= {"center": extra / 2, "right": extra, "end": extra}.get(item["align"], 0)
    multiple = line / (1.2 * size)                 # Google Slides spaces lines by a multiple of 1.2 em
    if google:
        # where the first baseline lands below the box top. Browser: the ascent plus half the leading
        # (line height minus ascent and descent). Google Slides fills a line of 1.2 em times the
        # multiple (never more than 1) in the ratio of ascent to descent; for Arial this gives the
        # 0.97 em and the raise of 0.97 em per unit below 1 that its PDF export showed.
        ascent, descent = METRICS.get(family, METRICS["Inter"])
        browser = ascent * size + (line - (ascent + descent) * size) / 2
        google_slides = 1.2 * min(multiple, 1.0) * ascent / (ascent + descent) * size
        y += browser - google_slides
    box = shapes.add_textbox(emu(x), emu(y), emu(width), emu(item["h"]))
    frame = box.text_frame
    frame.word_wrap = False
    frame.vertical_anchor = MSO_ANCHOR.TOP
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    paragraph = frame.paragraphs[0]
    paragraph.alignment = {"center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT, "end": PP_ALIGN.RIGHT}.get(
        item["align"], PP_ALIGN.LEFT)
    # PowerPoint and LibreOffice keep an exact line pitch; Google Slides needs the multiple
    paragraph.line_spacing = multiple if google else Pt(line / 2)
    for n, runs in enumerate(item["lines"]):
        if n:
            if google:                             # a bare break would count as 18 pt Calibri
                props = paragraph._p.add_br().get_or_add_rPr()
                props.set("sz", str(size * 50))
                etree.SubElement(props, qn("a:latin")).set("typeface", family)
            else:
                paragraph.add_line_break()
        for part in runs:
            run = paragraph.add_run()
            run.text = part["text"]
            run.font.name = family
            run.font.size = Pt(size / 2)
            run.font.bold = bool(part["bold"])
            run.font.italic = bool(part["italic"])
            run.font.color.rgb = colour(part["color"])
            # the same face for every script, so arrows and other symbols do not fall back to another face
            latin = run._r.get_or_add_rPr().find(qn("a:latin"))
            for tag in ("a:cs", "a:ea"):
                latin.addnext(etree.Element(qn(tag), typeface=family))
            if item["spacing"]:
                run._r.get_or_add_rPr().set("spc", str(round(item["spacing"] * 50)))
    return box


def timing_xml(clicks: list[list[int]]) -> etree._Element:
    """Main animation sequence: each click fades in one or more shapes (by shape id)."""
    counter = iter(range(3, 10_000))

    def effect(shape_id: int, first: bool) -> str:
        node = "clickEffect" if first else "withEffect"
        a, b, c = next(counter), next(counter), next(counter)
        return (f'<p:par><p:cTn id="{a}" presetID="10" presetClass="entr" presetSubtype="0" fill="hold" '
                f'grpId="0" nodeType="{node}"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
                f'<p:childTnLst><p:set><p:cBhvr><p:cTn id="{b}" dur="1" fill="hold"><p:stCondLst>'
                f'<p:cond delay="0"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{shape_id}"/></p:tgtEl>'
                f'<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr>'
                f'<p:to><p:strVal val="visible"/></p:to></p:set><p:animEffect transition="in" filter="fade">'
                f'<p:cBhvr><p:cTn id="{c}" dur="500"/><p:tgtEl><p:spTgt spid="{shape_id}"/></p:tgtEl></p:cBhvr>'
                f'</p:animEffect></p:childTnLst></p:cTn></p:par>')

    steps = []
    for shape_ids in clicks:
        outer, inner = next(counter), next(counter)
        effects = "".join(effect(sid, n == 0) for n, sid in enumerate(shape_ids))
        steps.append(f'<p:par><p:cTn id="{outer}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst>'
                     f'<p:childTnLst><p:par><p:cTn id="{inner}" fill="hold"><p:stCondLst><p:cond delay="0"/>'
                     f'</p:stCondLst><p:childTnLst>{effects}</p:childTnLst></p:cTn></p:par></p:childTnLst>'
                     f'</p:cTn></p:par>')
    xml = (f'<p:timing xmlns:p="{P_NS}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" '
           f'nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" '
           f'nodeType="mainSeq"><p:childTnLst>{"".join(steps)}</p:childTnLst></p:cTn><p:prevCondLst>'
           f'<p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst>'
           f'<p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
           f'</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')
    return etree.fromstring(xml)


def backup_ids(index: dict) -> set[str]:
    """The slides from the start of the deck's backup section to its end."""
    order, start = index["order"], index["sections"].get("backup", {}).get("start")
    return set(order[order.index(start):]) if start in order else set()


def build(out: Path = OUT, main_only: bool = False, google: bool = False) -> None:
    slides = measured_layout()
    unknown = sorted({i["family"] for s in slides for i in s["items"] if i["kind"] == "text"} - set(METRICS))
    if unknown:
        raise SystemExit(f"Text in a face this script has no metrics for: {', '.join(unknown)}")
    if main_only:
        skip = backup_ids(deck.load_index())
        slides = [slide for slide in slides if slide["id"] not in skip]
    prs = Presentation()
    prs.slide_width, prs.slide_height = emu(1920), emu(1080)
    blank = prs.slide_layouts[6]
    prs.core_properties.title = deck.load_index()["title"]
    prs.core_properties.author = prs.core_properties.last_modified_by = ""      # no names in the bundle

    for data in slides:
        slide = prs.slides.add_slide(blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = colour(data["background"])

        # shapes of one fragment share a group, so each click step is one or two entries in the timeline
        sizes: dict = {}
        for item in data["items"]:
            if item["fragment"]:
                parts = 1 if item["kind"] != "box" else (item["fill"] != "rgba(0, 0, 0, 0)") + len(item["edges"])
                sizes[item["fragment"]] = sizes.get(item["fragment"], 0) + parts
        groups, by_step = {}, {}
        for item in data["items"]:
            fragment = item["fragment"]
            target = slide.shapes
            if fragment and sizes[fragment] > 1:
                if fragment not in groups:
                    groups[fragment] = slide.shapes.add_group_shape()
                    by_step.setdefault(item["step"], []).append(groups[fragment].shape_id)
                target = groups[fragment].shapes
            if item["kind"] == "box":
                made = add_box(target, item)
            elif item["kind"] == "image":
                made = [add_image(target, item)]
            else:
                made = [add_text(target, item, google)]
            if fragment and sizes[fragment] == 1:
                by_step.setdefault(item["step"], []).extend(shape.shape_id for shape in made)

        root = slide._element
        transition = etree.SubElement(root, qn("p:transition"))
        transition.set("spd", "med")
        etree.SubElement(transition, qn("p:fade"))
        if by_step:
            root.append(timing_xml([by_step[step] for step in sorted(by_step)]))
        if data["notes"].strip():
            slide.notes_slide.notes_text_frame.text = data["notes"].strip()

    out = out.resolve()
    prs.save(out)
    steps = sum(len({i["step"] for i in s["items"] if i["step"]}) for s in slides)
    shown = out.relative_to(deck.SLIDES.parent) if out.is_relative_to(deck.SLIDES.parent) else out
    print(f"Wrote {shown} ({len(slides)} slides, {steps} click steps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--main-only", action="store_true", help="leave out the backup slides")
    parser.add_argument("--google", action="store_true", help="tune the text boxes for an upload to Google Slides")
    parser.add_argument("--out", type=Path, default=OUT, help=f"output file (default: {OUT.name} next to this script)")
    args = parser.parse_args()
    build(args.out, args.main_only, args.google)
