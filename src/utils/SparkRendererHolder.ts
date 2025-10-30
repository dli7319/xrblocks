import type {SparkRenderer} from '@sparkjsdev/spark';

// Object which just holds the spark renderer so other classes don't need to import spark.
export class SparkRendererHolder {
  constructor(public renderer: SparkRenderer) {}
}
