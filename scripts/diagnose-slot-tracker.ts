/**
 * WHAT DOES "tracker swap" ACTUALLY COUNT?
 *
 * Real-world word-mode telemetry reported 91 tracker swaps on a 93-frame
 * gesture -- a swap on essentially every frame -- which looks like the
 * left/right slots oscillating and corrupting the stored trajectory.
 *
 * This decides the question empirically, on REAL KArSL landmark motion, by
 * feeding the same physical hands to the live tracker under three different
 * MediaPipe ORDERINGS and measuring two things separately:
 *
 *   switchCount   how often the tracker reordered the incoming array
 *   slot purity   whether a slot actually kept hold of one physical hand
 *
 * Those are not the same quantity, and the whole confusion is that the
 * telemetry showed only the first. Ground truth is available because the
 * processed take already names each hand, so "which physical hand ended up in
 * slot 0" is checkable rather than inferred.
 *
 * DIAGNOSTIC ONLY. Nothing here is imported by the app, and this script
 * changes no behaviour.
 *
 * Usage: npx tsx scripts/diagnose-slot-tracker.ts [--take <processed json>]
 */
import { readFileSync } from "fs";
import { HandSlotTracker, type SlotDetection } from "../lib/ml/handSlotTracker";
import { assignSides } from "../lib/ml/handSides";
import type { HandLandmarks } from "../types/landmarks";

function arg(name: string, fallback: string): string {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

interface ProcessedHand { landmarks: HandLandmarks; rawLabel?: "Left" | "Right" }
interface ProcessedFrame { hands?: { left?: ProcessedHand | null; right?: ProcessedHand | null } }

/** One frame's two hands, with the truth of which is which kept alongside. */
interface TruthFrame {
    left: ProcessedHand | null;
    right: ProcessedHand | null;
}

type Ordering = "stable" | "inverted" | "alternating";

/**
 * Builds the detection array a live MediaPipe frame would hand us, under a
 * chosen ordering. The PHYSICAL hands are identical in all three cases -- only
 * the array order differs, which is exactly the variable under test.
 */
function detectionsFor(frame: TruthFrame, index: number, ordering: Ordering): SlotDetection[] {
    const pair: Array<{ hand: ProcessedHand; truth: "left" | "right" }> = [];
    if (frame.left) pair.push({ hand: frame.left, truth: "left" });
    if (frame.right) pair.push({ hand: frame.right, truth: "right" });
    if (pair.length < 2) {
        return pair.map((p) => ({ landmarks: p.hand.landmarks, handedness: p.hand.rawLabel ?? null, score: 0.9 }));
    }
    let order = pair;
    if (ordering === "inverted") order = [pair[1], pair[0]];
    if (ordering === "alternating" && index % 2 === 1) order = [pair[1], pair[0]];
    return order.map((p) => ({
        landmarks: p.hand.landmarks,
        handedness: p.hand.rawLabel ?? null,
        score: 0.9,
    }));
}

/** Which physical hand each slot holds this frame, or null. */
function truthOfSlots(
    frame: TruthFrame,
    slots: ReturnType<HandSlotTracker["step"]>
): [("left" | "right" | null), ("left" | "right" | null)] {
    const idOf = (d: SlotDetection | null): "left" | "right" | null => {
        if (!d) return null;
        if (frame.left && d.landmarks === frame.left.landmarks) return "left";
        if (frame.right && d.landmarks === frame.right.landmarks) return "right";
        return null;
    };
    return [idOf(slots[0]), idOf(slots[1])];
}

function run(frames: TruthFrame[], ordering: Ordering) {
    const tracker = new HandSlotTracker();
    const slot0Truth: Array<"left" | "right" | null> = [];
    const swapFlags: boolean[] = [];
    const slotWrists: Array<[number | null, number | null]> = [];
    let before = 0;

    for (let i = 0; i < frames.length; i++) {
        const slots = tracker.step(detectionsFor(frames[i], i, ordering));
        swapFlags.push(tracker.switchCount > before);
        before = tracker.switchCount;
        const [t0] = truthOfSlots(frames[i], slots);
        slot0Truth.push(t0);
        slotWrists.push([
            slots[0] ? slots[0]!.landmarks[0].x : null,
            slots[1] ? slots[1]!.landmarks[0].x : null,
        ]);
    }

    // SLOT PURITY: of the frames where slot 0 held a known hand, how often was
    // it the SAME physical hand as the majority? 1.0 means the slot never
    // changed identity -- which is the property the tracker exists to provide.
    const seen = slot0Truth.filter((t): t is "left" | "right" => t !== null);
    const lefts = seen.filter((t) => t === "left").length;
    const majority = Math.max(lefts, seen.length - lefts);
    const purity = seen.length ? majority / seen.length : 1;

    // Longest run of frames in which slot 0 held one physical hand without
    // changing, and how many times it DID change.
    let longest = 0, current = 0, identityChanges = 0;
    let prev: "left" | "right" | null = null;
    for (const t of slot0Truth) {
        if (t === null) continue;
        if (prev !== null && t !== prev) { identityChanges++; current = 0; }
        current++;
        if (current > longest) longest = current;
        prev = t;
    }

    return { ordering, swaps: tracker.switchCount, purity, longest, identityChanges, frames: frames.length, swapFlags, slotWrists, slot0Truth };
}

/** Per-frame geometry, with no temporal memory at all: signer's right hand is
 * at the smaller image x in an un-mirrored frame. */
function perFrameGeometry(frames: TruthFrame[]) {
    let agree = 0, compared = 0, flips = 0;
    let prev: "left" | "right" | null = null;
    for (const f of frames) {
        if (!f.left || !f.right) continue;
        compared++;
        // Which physical hand geometry would put on the signer's right.
        const truth: "left" | "right" = f.right.landmarks[0].x < f.left.landmarks[0].x ? "right" : "left";
        if (prev !== null && truth !== prev) flips++;
        prev = truth;
        if (truth === "right") agree++;
    }
    return { compared, agree, flips };
}

function main(): void {
    const takePath = arg("take", "processed-karsl/0071/03_01_0071_(01_12_16_15_53_03)_c.json");
    const raw = JSON.parse(readFileSync(takePath, "utf-8")) as { frames?: ProcessedFrame[] };
    const frames: TruthFrame[] = (raw.frames ?? []).map((f) => ({
        left: f.hands?.left ?? null,
        right: f.hands?.right ?? null,
    }));
    const twoHand = frames.filter((f) => f.left && f.right).length;

    const bar = "=".repeat(78);
    console.log(bar);
    console.log("SLOT TRACKER DIAGNOSTIC -- what does switchCount actually measure?");
    console.log(bar);
    console.log(`   take                 ${takePath}`);
    console.log(`   frames               ${frames.length}   (two-hand frames ${twoHand})`);
    console.log();
    console.log("   The SAME physical hands are fed in all three rows. Only MediaPipe's");
    console.log("   array ORDER differs. Slot purity 1.000 means no slot ever changed");
    console.log("   which physical hand it held.");
    console.log();
    console.log("   mediapipe order   swaps   swaps/frame   slot purity   identity changes   longest stable run");
    for (const ordering of ["stable", "inverted", "alternating"] as const) {
        const r = run(frames, ordering);
        console.log(
            `   ${ordering.padEnd(16)} ${String(r.swaps).padStart(5)}   ` +
            `${(r.swaps / r.frames).toFixed(3).padStart(11)}   ` +
            `${r.purity.toFixed(3).padStart(11)}   ` +
            `${String(r.identityChanges).padStart(16)}   ${String(r.longest).padStart(18)}`
        );
    }

    console.log();
    console.log("   NO TRACKER (trust MediaPipe's array order as the slot order):");
    for (const ordering of ["stable", "inverted", "alternating"] as const) {
        // Slot 0 = detections[0] verbatim, which is what "no tracker" means.
        let changes = 0;
        let prev: "left" | "right" | null = null;
        for (let i = 0; i < frames.length; i++) {
            const f = frames[i];
            if (!f.left || !f.right) continue;
            const det = detectionsFor(f, i, ordering);
            const truth = det[0].landmarks === f.left.landmarks ? "left" : "right";
            if (prev !== null && truth !== prev) changes++;
            prev = truth;
        }
        console.log(`   ${ordering.padEnd(16)} slot-0 identity changes: ${changes}`);
    }

    const geo = perFrameGeometry(frames);
    console.log();
    console.log("   PER-FRAME GEOMETRY (no temporal memory):");
    console.log(`   two-hand frames compared ${geo.compared}, geometry and stored naming agree on ${geo.agree}`);
    console.log(`   times the geometric answer FLIPPED between consecutive frames: ${geo.flips}`);

    const assignment = assignSides(
        frames.map((f) => f.left?.landmarks ?? null),
        frames.map((f) => f.right?.landmarks ?? null),
        frames.map((f) => f.left?.rawLabel ?? null),
        frames.map((f) => f.right?.rawLabel ?? null)
    );
    console.log();
    console.log(`   WHOLE-GESTURE assignSides(): sides=${JSON.stringify(assignment.sides)} ` +
        `source=${assignment.source} confidence=${assignment.confidence.toFixed(2)}`);
    console.log(bar);
}

main();
