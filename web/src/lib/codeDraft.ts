const DRAFT_KEY_PREFIX = 'pyre-code-draft:';

function getDraftKey(problemId: string) {
  return `${DRAFT_KEY_PREFIX}${problemId}`;
}

function isLegacyPpoFullLossDraft(code: string): boolean {
  const signature = code.match(/\bdef\s+ppo_full_loss\s*\(([\s\S]*?)\)\s*(?:->[^\n:]+)?\:/);
  const params = signature?.[1] ?? '';
  return (
    params.includes('old_values') ||
    params.includes('entropy') ||
    params.includes('value_coef') ||
    params.includes('entropy_coef') ||
    params.includes('value_clip_ratio')
  );
}

export function isStaleCodeDraft(problemId: string, code: string): boolean {
  return problemId === 'ppo_full_loss' && isLegacyPpoFullLossDraft(code);
}

export function loadCodeDraft(problemId: string): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage.getItem(getDraftKey(problemId));
  } catch {
    return null;
  }
}

export function saveCodeDraft(problemId: string, code: string) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(getDraftKey(problemId), code);
  } catch {
    // Ignore storage write failures (private mode, quota, etc).
  }
}
