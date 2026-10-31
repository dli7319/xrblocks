import * as xb from "xrblocks";
import { LitElement } from "lit";
export declare class EnterXRButton extends LitElement {
  static styles: import("lit").CSSResult;
  simulatorMode: xb.SimulatorMode;
  setSimulatorMode(newMode: xb.SimulatorMode): void;
  onClick(): void;
  render(): import("lit").TemplateResult<1>;
}