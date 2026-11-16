export type TurnEntity = {
  id: string;
  cls: string;
  cat: string;
  conf: number | null;
  /** Normalised 0..1 box, top-left origin. */
  x: number;
  y: number;
  w: number;
  h: number;
};

export type TurnData = {
  frame: { width: number; height: number; run: string };
  source: string;
  model: string | null;
  reviewed_boxes: number;
  classes: Record<string, number>;
  entities: TurnEntity[];
  /** Sprites placed by template matching (scripts/build_walkthrough_sprites.py). */
  world?: {
    /** Tile width in frame pixels. */
    tile: number;
    /** Frame pixel of the tile grid's origin. */
    origin: number[];
    items: { id: string; sprite: string; x: number; y: number; score: number }[];
  };
  hud: {
    readings: Record<string, string | number>;
    /** Normalised [x0, y0, x1, y1] per HUD field. */
    fields: Record<string, number[]>;
  };
};
