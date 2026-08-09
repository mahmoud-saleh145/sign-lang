/**
 * Abstraction over speech-to-text so the UI never talks to
 * webkitSpeechRecognition directly. Swapping to a production speech
 * service later only requires a new class implementing this interface.
 */
export interface SpeechRecognitionService {
  start(): void;
  stop(): void;
  onResult(callback: (text: string, isFinal: boolean) => void): void;
  onError(callback: (message: string) => void): void;
  onEnd(callback: () => void): void;
  isSupported(): boolean;
}

// The Web Speech API's TS types are inconsistent across lib.dom versions,
// so we declare the minimal shape we actually use rather than pulling in `any`.
interface WebSpeechAlternative {
  transcript: string;
}
interface WebSpeechResult {
  0: WebSpeechAlternative;
  isFinal: boolean;
}
interface WebSpeechResultList {
  length: number;
  [index: number]: WebSpeechResult;
}
interface WebSpeechEvent {
  resultIndex: number;
  results: WebSpeechResultList;
}
interface WebSpeechErrorEvent {
  error: string;
}
interface WebSpeechRecognition {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  onresult: ((event: WebSpeechEvent) => void) | null;
  onerror: ((event: WebSpeechErrorEvent) => void) | null;
  onend: (() => void) | null;
}
interface WebSpeechRecognitionConstructor {
  new (): WebSpeechRecognition;
}
interface WindowWithSpeech extends Window {
  SpeechRecognition?: WebSpeechRecognitionConstructor;
  webkitSpeechRecognition?: WebSpeechRecognitionConstructor;
}

export class BrowserSpeechRecognitionService implements SpeechRecognitionService {
  private recognition: WebSpeechRecognition | null = null;
  private resultCallback: ((text: string, isFinal: boolean) => void) | null = null;
  private errorCallback: ((message: string) => void) | null = null;
  private endCallback: (() => void) | null = null;

  constructor(lang: string = "ar-EG") {
    if (typeof window === "undefined") return;
    const w = window as WindowWithSpeech;
    const Ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition;
    if (!Ctor) return;

    this.recognition = new Ctor();
    this.recognition.lang = lang;
    this.recognition.continuous = true;
    this.recognition.interimResults = true;

    this.recognition.onresult = (event: WebSpeechEvent) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        this.resultCallback?.(result[0].transcript, result.isFinal);
      }
    };
    this.recognition.onerror = (event: WebSpeechErrorEvent) => {
      this.errorCallback?.(mapSpeechError(event.error));
    };
    this.recognition.onend = () => {
      this.endCallback?.();
    };
  }

  isSupported(): boolean {
    return this.recognition !== null;
  }

  start(): void {
    if (!this.recognition) {
      this.errorCallback?.("Speech recognition is not supported in this browser.");
      return;
    }
    try {
      this.recognition.start();
    } catch {
      // start() throws if already running; ignore, matches native semantics.
    }
  }

  stop(): void {
    this.recognition?.stop();
  }

  onResult(callback: (text: string, isFinal: boolean) => void): void {
    this.resultCallback = callback;
  }

  onError(callback: (message: string) => void): void {
    this.errorCallback = callback;
  }

  onEnd(callback: () => void): void {
    this.endCallback = callback;
  }
}

function mapSpeechError(code: string): string {
  switch (code) {
    case "not-allowed":
    case "permission-denied":
      return "Microphone permission was denied.";
    case "no-speech":
      return "No speech was detected.";
    case "audio-capture":
      return "No microphone was found.";
    case "network":
      return "A network error interrupted speech recognition.";
    default:
      return `Speech recognition error: ${code}`;
  }
}
