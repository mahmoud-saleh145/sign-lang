import {
  HandLandmarker,
  FilesetResolver,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import type { FrameDetection, DetectedHand, Handedness } from "@/types/landmarks";

const WASM_BASE_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_ASSET_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export interface HandLandmarkerServiceOptions {
  numHands?: number;
  minHandDetectionConfidence?: number;
  minHandPresenceConfidence?: number;
  minTrackingConfidence?: number;
}

/**
 * Thin wrapper around MediaPipe's HandLandmarker for live video streams.
 * Must be created and used only in the browser (client components).
 */
export class HandLandmarkerService {
  private landmarker: HandLandmarker | null = null;

  private constructor(landmarker: HandLandmarker) {
    this.landmarker = landmarker;
  }

  static async create(options: HandLandmarkerServiceOptions = {}): Promise<HandLandmarkerService> {
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
    const landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_ASSET_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: options.numHands ?? 1,
      minHandDetectionConfidence: options.minHandDetectionConfidence ?? 0.6,
      minHandPresenceConfidence: options.minHandPresenceConfidence ?? 0.6,
      minTrackingConfidence: options.minTrackingConfidence ?? 0.6,
    });
    return new HandLandmarkerService(landmarker);
  }

  /** Run detection on one video frame. Caller drives the frame loop (rAF). */
  detectForVideo(video: HTMLVideoElement, timestampMs: number): FrameDetection {
    if (!this.landmarker) {
      throw new Error("HandLandmarkerService used after close()");
    }
    const result: HandLandmarkerResult = this.landmarker.detectForVideo(video, timestampMs);

    const hands: DetectedHand[] = result.landmarks.map((landmarks, i) => {
      const handednessInfo = result.handedness[i]?.[0];
      return {
        landmarks: landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z })),
        handedness: (handednessInfo?.categoryName as Handedness) ?? "Right",
        score: handednessInfo?.score ?? 0,
      };
    });

    return { timestampMs, hands };
  }

  close(): void {
    this.landmarker?.close();
    this.landmarker = null;
  }
}
