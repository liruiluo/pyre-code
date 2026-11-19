'use client';

import { Bookmark, BookmarkPlus } from 'lucide-react';
import { useLocale } from '@/context/LocaleContext';
import { useStudyLists } from '@/context/StudyListsContext';

interface StudyListPickerButtonProps {
  problemId: string;
  problemTitle: string;
  compact?: boolean;
}

export function StudyListPickerButton({
  problemId, problemTitle, compact = false,
}: StudyListPickerButtonProps) {
  const { t } = useLocale();
  const { lists, loading, openPicker } = useStudyLists();
  const count = lists.filter((list) => list.problemIds.includes(problemId)).length;
  const label = count ? t('savedInLists', { n: count }) : t('addToList');

  return (
    <button
      type="button"
      disabled={loading}
      title={label}
      aria-label={label + ': ' + problemTitle}
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        openPicker({ id: problemId, title: problemTitle });
      }}
      className="inline-flex items-center justify-center gap-1.5 h-8 px-2.5 rounded-lg text-[12px] text-text-2 hover:text-accent disabled:opacity-40 transition-colors"
      style={{ background: 'var(--bg-elev)', border: '1px solid var(--line)' }}
    >
      {count ? (
        <Bookmark className="w-3.5 h-3.5 text-accent" fill="currentColor" />
      ) : (
        <BookmarkPlus className="w-3.5 h-3.5" />
      )}
      {!compact && <span>{label}</span>}
      {compact && count > 0 && <span className="mono text-[10px] text-accent">{count}</span>}
    </button>
  );
}
