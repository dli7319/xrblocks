import { Signal } from "@preact/signals-core";
export type ButtonProperties = {
  text: string | Signal<string>;
  icon?: string | Signal<string>;
};