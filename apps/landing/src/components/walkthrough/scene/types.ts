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
  hud: {
    readings: Record<string, string | number>;
    /** Normalised [x0, y0, x1, y1] per HUD field. */
    fields: Record<string, number[]>;
  };
};
