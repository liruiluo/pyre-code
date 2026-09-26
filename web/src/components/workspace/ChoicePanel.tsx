'use client';

import { Check, X, ListChecks } from 'lucide-react';
import { useLocale } from '@/context/LocaleContext';
import type { Problem, SubmissionResult } from '@/lib/types';

interface ChoicePanelProps {
  problem: Problem & { starterCode?: string };
  selected: number | null;
  onSelect: (i: number) => void;
  onSubmit: () => void;
  onRetry: () => void;
  isSubmitting: boolean;
  result: SubmissionResult | null;
}

const LETTERS = ['A', 'B', 'C', 'D', 'E', 'F'];

export function ChoicePanel({ problem, selected, onSelect, onSubmit, onRetry, isSubmitting, result }: ChoicePanelProps) {
  const { locale, t } = useLocale();
  const options = problem.options ?? [];
  const answered = result !== null;
  const correct = result?.allPassed ?? false;
  const answerIndex = (result?.answerIndex ?? null) as number | null;
  const explanation = locale === 'zh' ? problem.explanationZh : problem.explanationEn;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div
        className="flex items-center gap-2.5 px-3.5 flex-shrink-0"
        style={{
          height: '36px',
          borderBottom: '1px solid var(--line)',
          background: 'color-mix(in oklab, var(--text) 2%, var(--bg))',
        }}
      >
        <ListChecks className="w-3.5 h-3.5 text-text-3" />
        <span className="mono text-[12.5px] text-text-2">{problem.id}</span>
        <span
          className="ml-auto mono text-[11.5px] text-text-3 px-1.5 py-0.5 rounded"
          style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
        >
          {t('multipleChoice')}
        </span>
      </div>

      {/* Options */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-3 max-w-[760px]">
        {options.map((option, i) => {
          const isSelected = selected === i;
          const isAnswer = answered && answerIndex !== null && i === answerIndex;
          const isWrongPick = answered && isSelected && !correct;
          return (
            <button
              key={i}
              disabled={answered}
              onClick={() => onSelect(i)}
              className="w-full flex items-start gap-3.5 text-left px-4 py-3.5 rounded-[10px] transition-[border-color,background] duration-150 cursor-pointer disabled:cursor-default"
              style={{
                border: isAnswer
                  ? '1px solid var(--easy)'
                  : isWrongPick
                    ? '1px solid var(--medium)'
                    : isSelected
                      ? '1px solid var(--accent)'
                      : '1px solid var(--line)',
                background: isAnswer
                  ? 'color-mix(in oklab, var(--easy) 8%, var(--bg-elev))'
                  : isWrongPick
                    ? 'color-mix(in oklab, var(--medium) 8%, var(--bg-elev))'
                    : isSelected
                      ? 'var(--accent-wash)'
                      : 'var(--bg-elev)',
              }}
            >
              <span
                className="mono text-[11.5px] w-6 h-6 flex-shrink-0 inline-flex items-center justify-center rounded-[6px] mt-px"
                style={{
                  border: `1px solid ${isAnswer ? 'var(--easy)' : isWrongPick ? 'var(--medium)' : isSelected ? 'var(--accent)' : 'var(--line)'}`,
                  color: isAnswer ? 'var(--easy)' : isWrongPick ? 'var(--medium)' : isSelected ? 'var(--accent)' : 'var(--text-3)',
                }}
              >
                {LETTERS[i] ?? i + 1}
              </span>
              <span className="text-sm text-text leading-relaxed flex-1">{option}</span>
              {isAnswer && <Check className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: 'var(--easy)' }} />}
              {isWrongPick && <X className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: 'var(--medium)' }} />}
            </button>
          );
        })}

        {/* Result banner + explanation */}
        {answered && (
          <div
            className="mt-5 p-4 rounded-[10px]"
            style={{
              background: correct
                ? 'color-mix(in oklab, var(--easy) 6%, var(--bg-elev))'
                : 'color-mix(in oklab, var(--medium) 6%, var(--bg-elev))',
              border: `1px solid ${correct ? 'color-mix(in oklab, var(--easy) 35%, var(--line))' : 'color-mix(in oklab, var(--medium) 35%, var(--line))'}`,
            }}
          >
            <div
              className="mono text-[11px] tracking-[0.12em] uppercase font-semibold mb-2"
              style={{ color: correct ? 'var(--easy)' : 'var(--medium)' }}
            >
              {correct ? t('correctAnswer') : t('wrongAnswer')}
            </div>
            {explanation && (
              <div className="space-y-1.5">
                <div className="mono text-[10.5px] tracking-[0.12em] uppercase text-text-3 font-semibold">
                  ⚑ {t('explanationTitle')}
                </div>
                {explanation.split('\n').map((line, i) => (
                  <p key={i} className="text-sm text-text-2 leading-relaxed">{line}</p>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Submit bar */}
      <div
        className="flex items-center gap-3 px-5 py-3 flex-shrink-0"
        style={{ borderTop: '1px solid var(--line)', background: 'var(--bg)' }}
      >
        <span className="text-[12.5px] text-text-3">{!answered && (selected === null ? t('selectAnOption') : '')}</span>
        {answered && !correct ? (
          <button
            onClick={onRetry}
            className="ml-auto h-[34px] px-5 rounded-[8px] text-[13px] font-medium cursor-pointer"
            style={{ border: '1px solid var(--accent-line)', background: 'var(--accent-wash)', color: 'var(--accent)' }}
          >
            {locale === 'zh' ? '重试' : 'Try Again'}
          </button>
        ) : (
          <button
            onClick={onSubmit}
            disabled={isSubmitting || selected === null || answered}
            className="ml-auto h-[34px] px-5 rounded-[8px] text-[13px] font-medium text-white cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed transition-[opacity] duration-150"
            style={{ background: 'var(--accent)' }}
          >
            {isSubmitting ? t('judging') : t('submitAnswer')}
          </button>
        )}
      </div>
    </div>
  );
}
