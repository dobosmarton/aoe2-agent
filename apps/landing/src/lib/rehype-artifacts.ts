/**
 * Rehype plugin: puts every table, code block, mermaid diagram and lone image in the
 * docs inside the detector frame that the blog uses for its artifacts (see
 * `.artifact` in styles/globals.css and components/ui/artifact-frame.tsx):
 *
 *   <figure class="artifact">
 *     <div class="artifact-frame">
 *       <svg class="artifact-corner ..."/> ×4     "+" registration marks
 *       ...the artifact...
 *     </div>
 *     <figcaption class="artifact-caption">LABEL <span class="artifact-label-meta">META</span></figcaption>
 *   </figure>
 *
 * Labels follow the blog: a table is named by its first header cell with the row
 * count as meta, a code block is "Code" with its line count, and an image uses its
 * alt text and file format. Code blocks also get a header strip with the language
 * and a copy button (the click handler lives in layouts/Site.astro).
 *
 * Runs after Shiki and rehype-mermaid, so it sees `pre.astro-code` and the inline
 * mermaid `<svg>`.
 */
import { SKIP, visit } from "unist-util-visit";

// The few hast shapes this plugin touches. Typed locally so the plugin does not
// depend on @types/hast being hoisted into this workspace.
type Text = { type: "text"; value: string };
type Element = {
  type: "element";
  tagName: string;
  properties: Record<string, unknown>;
  children: Node[];
};
type Node = Element | Text | { type: string; children?: Node[]; value?: string };
type Parent = { type: string; children: Node[] };

const LANGUAGE_LABELS: Record<string, string> = { plaintext: "text", txt: "text", sh: "bash", shell: "bash" };

function el(tagName: string, properties: Record<string, unknown>, children: Node[] = []): Element {
  return { type: "element", tagName, properties, children };
}

function text(value: string): Text {
  return { type: "text", value };
}

function isElement(node: Node | undefined, tagName?: string): node is Element {
  return node?.type === "element" && (tagName === undefined || (node as Element).tagName === tagName);
}

function classes(node: Element): string[] {
  // Parsed HTML has `className`; Shiki's hast output has the raw `class` string.
  const value = node.properties.className ?? node.properties.class;
  if (Array.isArray(value)) return value.map(String);
  if (typeof value === "string") return value.split(/\s+/);
  return [];
}

function textOf(node: Node): string {
  if (node.type === "text") return (node as Text).value;
  return (node.children ?? []).map(textOf).join("");
}

function corner(position: "tl" | "tr" | "bl" | "br"): Element {
  return el(
    "svg",
    { className: ["artifact-corner", `artifact-corner-${position}`], viewBox: "0 0 7 7", ariaHidden: "true" },
    [el("path", { d: "M3.5 0v7M0 3.5h7", stroke: "currentColor", strokeWidth: "1" })],
  );
}

function artifact(content: Node[], label: string, meta?: string): Element {
  const caption: Node[] = [text(label)];
  if (meta) caption.push(el("span", { className: ["artifact-label-meta"] }, [text(` ${meta}`)]));
  return el("figure", { className: ["artifact"] }, [
    el("div", { className: ["artifact-frame"] }, [corner("tl"), corner("tr"), corner("bl"), corner("br"), ...content]),
    el("figcaption", { className: ["artifact-caption"] }, caption),
  ]);
}

const plural = (count: number, word: string) => `${count} ${word}${count === 1 ? "" : "s"}`;

function wrapTable(table: Element): Element {
  const head = table.children.find((child) => isElement(child, "thead")) as Element | undefined;
  const body = table.children.find((child) => isElement(child, "tbody")) as Element | undefined;
  const firstRow = head?.children.find((child) => isElement(child, "tr")) as Element | undefined;
  const firstCell = firstRow?.children.find((child) => isElement(child, "th")) as Element | undefined;
  const label = (firstCell ? textOf(firstCell).trim() : "") || "Table";
  const rows = body ? body.children.filter((child) => isElement(child, "tr")).length : 0;
  return artifact([el("div", { className: ["table-scroll"] }, [table])], label, plural(rows, "row"));
}

function wrapCode(pre: Element): Element {
  // Shiki writes the raw attribute name; hast parsers write the camel-cased property.
  const language = String(pre.properties.dataLanguage ?? pre.properties["data-language"] ?? "plaintext");
  const code = textOf(pre).replace(/\n$/, "");
  const lines = code.split("\n").length;
  const header = el("div", { className: ["cell-header"] }, [
    el("span", { className: ["bracket-label"] }, [text(LANGUAGE_LABELS[language] ?? language)]),
    el("span", { className: ["cell-aside"] }, [
      el("button", { type: "button", className: ["code-copy"], dataCopyCode: "", ariaLabel: "Copy code" }, [
        text("Copy"),
      ]),
    ]),
  ]);
  return artifact([el("div", { className: ["code-cell"] }, [header, pre])], "Code", plural(lines, "line"));
}

function wrapDiagram(svg: Element): Element {
  return artifact([el("div", { className: ["diagram-cell"] }, [svg])], "Diagram", "Mermaid");
}

function imageFormat(src: unknown): string | undefined {
  return String(src ?? "")
    .match(/\.([a-z0-9]+)(?:[?#].*)?$/i)?.[1]
    ?.toUpperCase();
}

// A paragraph that holds one image and nothing else becomes a framed figure.
function loneImage(paragraph: Element): Element | undefined {
  const content = paragraph.children.filter((child) => !(child.type === "text" && !textOf(child).trim()));
  const only = content[0];
  return content.length === 1 && isElement(only, "img") ? only : undefined;
}

export function rehypeArtifacts() {
  return (tree: Parent) => {
    visit(tree as never, "element", (node: Element, index: number | undefined, parent: Parent | undefined) => {
      if (!parent || index === undefined) return;
      if (isElement(parent as Node, "div") && classes(parent as Element).some((c) => c.endsWith("-cell") || c === "table-scroll")) {
        return SKIP;
      }

      let wrapped: Element | undefined;
      if (node.tagName === "table") {
        wrapped = wrapTable(node);
      } else if (node.tagName === "pre" && classes(node).includes("astro-code")) {
        wrapped = wrapCode(node);
      } else if (node.tagName === "svg" && String(node.properties.id ?? "").startsWith("mermaid")) {
        wrapped = wrapDiagram(node);
      } else if (node.tagName === "p") {
        const image = loneImage(node);
        if (image) {
          const alt = String(image.properties.alt ?? "").trim() || "Image";
          wrapped = artifact([image], alt, imageFormat(image.properties.src));
        }
      }

      if (!wrapped) return;
      parent.children[index] = wrapped;
      return SKIP;
    });
  };
}
