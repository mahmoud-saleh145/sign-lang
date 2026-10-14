/**
 * WHICH PROCESSED TREE THE PIPELINE READS, defined once.
 *
 * This exists because it was NOT defined once. The corpus was re-extracted
 * into processed-karsl-v2, the prototypes were rebuilt from it, and the
 * benchmark went on reading the old processed-karsl because its path was a
 * string literal in a different file. The run that followed scored old-tree
 * queries against new-tree prototypes -- 70 of the queried signs did not
 * exist in the registry at all -- and reported 0.3533 top-1 as though it
 * meant something. The real figure on a consistent corpus was 0.7558.
 *
 * Nothing about that was visible in the output, which is the point: a default
 * path repeated across five files is the same failure mode as a split rule
 * repeated across two, and it gets the same fix.
 *
 * Override for one run with KARSL_PROCESSED, or per script with --processed.
 */
export const DEFAULT_PROCESSED_ROOT = process.env.KARSL_PROCESSED ?? "processed-karsl-v2";

/**
 * Resolves a script's --processed argument against the shared default.
 * Pass the raw argument value (or null/undefined when it was not given).
 */
export function processedRoot(argValue?: string | null): string {
    return argValue && argValue.length > 0 ? argValue : DEFAULT_PROCESSED_ROOT;
}
