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
type PropertyValue = string | number | boolean | (string | number)[] | null | undefined;
type Properties = Record<string, PropertyValue>;
type Element = {
  type: "element";
  tagName: string;
  properties: Properties;
  children: Node[];
};
type Node = Element | Text | { type: string; children?: Node[]; value?: string };
type Parent = { type: string; children: Node[] };
type CornerPosition = "tl" | "tr" | "bl" | "br";

const LANGUAGE_LABELS: Record<string, string> = { plaintext: "text", txt: "text", sh: "bash", shell: "bash" };

const el = (tagName: string, properties: Properties, children: Node[] = []): Element => ({
  type: "element",
  tagName,
  properties,
  children,
});

const text = (value: string): Text => ({ type: "text", value });

const isText = (node: Node): node is Text => node.type === "text";

const isElement = (node: Node | undefined, tagName?: string): node is Element =>
  node?.type === "element" && "tagName" in node && (tagName === undefined || node.tagName === tagName);

const isBlankText = (node: Node): boolean => isText(node) && !node.value.trim();

const classes = (node: Element): string[] => {
  // Parsed HTML has `className`; Shiki's hast output has the raw `class` string.
  const value = node.properties.className ?? node.properties.class;

  if (Array.isArray(value)) {
    return value.map(String);
  }

  if (typeof value === "string") {
    return value.split(/\s+/);
  }

  return [];
};

const textOf = (node: Node): string => {
  if (isText(node)) {
    return node.value;
  }

  return (node.children ?? []).map(textOf).join("");
};

const corner = (position: CornerPosition): Element =>
  el("svg", { className: ["artifact-corner", `artifact-corner-${position}`], viewBox: "0 0 7 7", ariaHidden: "true" }, [
    el("path", { d: "M3.5 0v7M0 3.5h7", stroke: "currentColor", strokeWidth: "1" }),
  ]);

const artifact = (content: Node[], label: string, meta?: string): Element => {
  const caption: Node[] = [text(label)];

  if (meta) {
    caption.push(el("span", { className: ["artifact-label-meta"] }, [text(` ${meta}`)]));
  }

  return el("figure", { className: ["artifact"] }, [
    el("div", { className: ["artifact-frame"] }, [corner("tl"), corner("tr"), corner("bl"), corner("br"), ...content]),
    el("figcaption", { className: ["artifact-caption"] }, caption),
  ]);
};

const plural = (count: number, word: string): string => `${count} ${word}${count === 1 ? "" : "s"}`;

// The meta of a code artifact, "12 lines". Also used by components/CodeFigure.astro.
export const codeLinesMeta = (code: string): string => plural(code.replace(/\n$/, "").split("\n").length, "line");

const findElement = (children: readonly Node[], tagName: string): Element | undefined =>
  children.find((child) => isElement(child, tagName));

const wrapTable = (table: Element): Element => {
  const head = findElement(table.children, "thead");
  const body = findElement(table.children, "tbody");
  const firstRow = head && findElement(head.children, "tr");
  const firstCell = firstRow && findElement(firstRow.children, "th");
  const label = (firstCell ? textOf(firstCell).trim() : "") || "Table";
  const rows = body ? body.children.filter((child) => isElement(child, "tr")).length : 0;

  return artifact([el("div", { className: ["table-scroll"] }, [table])], label, plural(rows, "row"));
};

const wrapCode = (pre: Element): Element => {
  // Shiki writes the raw attribute name; hast parsers write the camel-cased property.
  const language = String(pre.properties.dataLanguage ?? pre.properties["data-language"] ?? "plaintext");
  const header = el("div", { className: ["cell-header"] }, [
    el("span", { className: ["bracket-label"] }, [text(LANGUAGE_LABELS[language] ?? language)]),
    el("span", { className: ["cell-aside"] }, [
      el("button", { type: "button", className: ["code-copy"], dataCopyCode: "", ariaLabel: "Copy code" }, [
        text("Copy"),
      ]),
    ]),
  ]);

  return artifact([el("div", { className: ["code-cell"] }, [header, pre])], "Code", codeLinesMeta(textOf(pre)));
};

const wrapDiagram = (svg: Element): Element => artifact([el("div", { className: ["diagram-cell"] }, [svg])], "Diagram", "Mermaid");

const imageFormat = (src: PropertyValue): string | undefined =>
  String(src ?? "")
    .match(/\.([a-z0-9]+)(?:[?#].*)?$/i)?.[1]
    ?.toUpperCase();

const wrapImage = (image: Element): Element => {
  const alt = String(image.properties.alt ?? "").trim() || "Image";

  return artifact([image], alt, imageFormat(image.properties.src));
};

// A paragraph that holds one image and nothing else becomes a framed figure.
const loneImage = (paragraph: Element): Element | undefined => {
  const content = paragraph.children.filter((child) => !isBlankText(child));
  const only = content[0];

  return content.length === 1 && isElement(only, "img") ? only : undefined;
};

const isMermaidDiagram = (node: Element): boolean => node.tagName === "svg" && String(node.properties.id ?? "").startsWith("mermaid");

const wrapArtifact = (node: Element): Element | undefined => {
  if (node.tagName === "table") {
    return wrapTable(node);
  }

  if (node.tagName === "pre" && classes(node).includes("astro-code")) {
    return wrapCode(node);
  }

  if (isMermaidDiagram(node)) {
    return wrapDiagram(node);
  }

  if (node.tagName === "p") {
    const image = loneImage(node);

    return image && wrapImage(image);
  }

  return undefined;
};

// Whatever already sits inside a code, diagram or table cell stays as it is.
const isInsideCell = (parent: Parent): boolean =>
  isElement(parent, "div") && classes(parent).some((name) => name.endsWith("-cell") || name === "table-scroll");

type Transformer = (tree: Parent) => void;

export const rehypeArtifacts = (): Transformer => (tree) => {
  visit(tree, "element", (node: Element, index: number | undefined, parent: Parent | undefined) => {
    if (!parent || index === undefined) {
      return;
    }

    if (isInsideCell(parent)) {
      return SKIP;
    }

    const wrapped = wrapArtifact(node);

    if (!wrapped) {
      return;
    }

    parent.children[index] = wrapped;

    return SKIP;
  });
};
