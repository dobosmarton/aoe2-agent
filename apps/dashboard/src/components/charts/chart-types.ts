// Shared chart types — the recharts integration boundary. Panels describe
// their series/data in terms of theme tokens (colorVar = "var(--food)") and
// never import recharts directly. Kept in a non-component module so the chart,
// legend, and tooltip files can share them without import cycles.

/** The string keys of `Row` whose values are `number` — the only columns a
 * chart series can legitimately plot. For the default dynamic row
 * (`Record<string, number>`) this collapses to `string`. */
export type NumberKeys<Row> = Extract<
  { [K in keyof Row]: Row[K] extends number ? K : never }[keyof Row],
  string
>;

/** The presentational shape of a chart series (key/label/color), independent of
 * the row type — what the legend and tooltip consume. Using this non-generic
 * supertype there sidesteps variance issues with `ChartSeries<Row>`. */
export type ChartSeriesBase = {
  /** Key into each data row. */
  readonly key: string;
  /** Human label shown in the tooltip/legend. */
  readonly label: string;
  /** A CSS color, typically a token: "var(--food)". */
  readonly colorVar: string;
}

/** A chart series whose `key` is narrowed to `Row`'s numeric columns, so a
 * series can only point at a real numeric chart column. */
export type ChartSeries<Row = Record<string, number>> = {
  readonly key: NumberKeys<Row>;
} & ChartSeriesBase

export type ComparisonDatum = {
  /** Category label on the X axis (e.g. a profile name or short run_id). */
  readonly name: string;
  readonly value: number;
  /** Render this bar at full opacity to mark it as the best/winner. */
  readonly highlight?: boolean;
}
