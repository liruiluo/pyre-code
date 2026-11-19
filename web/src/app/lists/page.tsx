'use client';

import { useEffect, useMemo, useState, type FormEvent } from 'react';
import Link from 'next/link';
import { Bookmark, Pencil, Plus, Trash2, X } from 'lucide-react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { StatusIcon } from '@/components/problem/StatusIcon';
import { Badge } from '@/components/ui/Badge';
import { StudyListPickerButton } from '@/components/study/StudyListPickerButton';
import { useStudyLists } from '@/context/StudyListsContext';
import { useLocale } from '@/context/LocaleContext';
import type { Problem, ProgressMap } from '@/lib/types';

const API_ROOT = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export default function MyListsPage() {
  const { locale, t } = useLocale();
  const { lists, loading, saving, error, mutate } = useStudyLists();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [newName, setNewName] = useState('');
  const [editing, setEditing] = useState(false);
  const [renameValue, setRenameValue] = useState('');
  const [problems, setProblems] = useState<Problem[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [catalogError, setCatalogError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(API_ROOT + '/api/problems').then((r) => {
        if (!r.ok) throw new Error('Cannot load problems');
        return r.json();
      }),
      fetch(API_ROOT + '/api/progress').then((r) => r.json()),
    ]).then(([catalog, done]) => {
      setProblems(catalog.problems ?? []);
      setProgress(done.progress ?? {});
    }).catch(() => setCatalogError(t('networkError')));
  }, []); // Load catalog and grading progress once.

  const activeList = lists.find((list) => list.id === selectedId) ?? lists[0] ?? null;
  const problemMap = useMemo(
    () => new Map(problems.map((problem) => [problem.id, problem])),
    [problems],
  );
  const listedProblems = (activeList?.problemIds ?? [])
    .map((id) => problemMap.get(id))
    .filter((problem): problem is Problem => problem !== undefined);
  const solvedCount = listedProblems.filter((problem) => progress[problem.id]?.status === 'solved').length;

  const createList = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = newName.trim();
    if (!name || saving) return;
    const existingIds = new Set(lists.map((list) => list.id));
    const updated = await mutate({ action: 'create', name });
    if (updated) {
      const created = updated.find((list) => !existingIds.has(list.id));
      setSelectedId(created?.id ?? selectedId);
      setNewName('');
    }
  };

  const renameList = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!activeList || !renameValue.trim() || saving) return;
    const updated = await mutate({
      action: 'rename',
      listId: activeList.id,
      name: renameValue.trim(),
    });
    if (updated) setEditing(false);
  };

  const deleteList = async () => {
    if (!activeList || saving || !window.confirm(t('deleteListConfirm'))) return;
    const updated = await mutate({ action: 'delete', listId: activeList.id });
    if (updated) {
      setSelectedId(updated[0]?.id ?? null);
      setEditing(false);
    }
  };

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="max-w-[1280px] mx-auto px-7 pt-10 pb-20">
        <div className="mb-10 pb-7" style={{ borderBottom: '1px solid var(--line)' }}>
          <div className="eyebrow mb-2.5">{t('myLists')}</div>
          <h1 className="text-[clamp(32px,4vw,52px)] font-semibold tracking-[-0.032em] leading-[1.05] mb-3.5">
            {t('myLists')}
          </h1>
          <p className="text-base text-text-2 max-w-[60ch]">{t('myListsSubtitle')}</p>
        </div>

        <div className="grid grid-cols-[290px_minmax(0,1fr)] gap-9 max-[850px]:grid-cols-1">
          <aside className="min-w-0">
            <div className="flex items-center gap-2 mb-4">
              <Bookmark className="w-4 h-4 text-accent" />
              <h2 className="font-semibold text-sm">{t('myLists')}</h2>
              <span className="ml-auto mono text-xs text-text-3">{lists.length}</span>
            </div>
            <div className="flex flex-col gap-2 mb-5">
              {loading && <p className="text-sm text-text-3">{t('loading')}</p>}
              {lists.map((list) => (
                <button
                  key={list.id}
                  type="button"
                  onClick={() => { setSelectedId(list.id); setEditing(false); }}
                  className="text-left px-4 py-3 rounded-xl transition-colors flex gap-3 items-center"
                  style={{
                    background: activeList?.id === list.id ? 'var(--accent-wash)' : 'var(--bg-elev)',
                    border: activeList?.id === list.id ? '1px solid var(--accent)' : '1px solid var(--line)',
                  }}
                >
                  <span className="flex-1 min-w-0 truncate text-sm font-medium">{list.name}</span>
                  <span className="mono text-xs text-text-3">{list.problemIds.length}</span>
                </button>
              ))}
            </div>
            <form onSubmit={(event) => void createList(event)} className="flex gap-2">
              <input
                value={newName}
                maxLength={40}
                onChange={(event) => setNewName(event.target.value)}
                placeholder={t('listNamePlaceholder')}
                aria-label={t('listNamePlaceholder')}
                className="min-w-0 flex-1 px-3 h-9 text-sm rounded-lg bg-bg-elev text-text outline-none focus:border-accent"
                style={{ border: '1px solid var(--line)' }}
              />
              <button
                type="submit"
                disabled={!newName.trim() || saving || loading}
                title={t('createList')}
                className="h-9 px-3 inline-flex items-center gap-1 rounded-lg text-sm bg-text text-bg disabled:opacity-40"
              >
                <Plus className="w-4 h-4" /> {t('createList')}
              </button>
            </form>
            {error && <p role="alert" className="mt-3 text-sm text-hard">{error}</p>}
          </aside>

          <section className="min-w-0">
            {activeList ? (
              <>
                <div className="flex items-center gap-3 flex-wrap mb-5 pb-5" style={{ borderBottom: '1px solid var(--line)' }}>
                  {editing ? (
                    <form onSubmit={(event) => void renameList(event)} className="flex items-center gap-2 flex-1">
                      <input
                        value={renameValue}
                        maxLength={40}
                        onChange={(event) => setRenameValue(event.target.value)}
                        aria-label={t('renameList')}
                        className="flex-1 min-w-0 h-9 px-3 rounded-lg text-sm bg-bg-elev outline-none"
                        style={{ border: '1px solid var(--accent)' }}
                        autoFocus
                      />
                      <button type="submit" disabled={!renameValue.trim() || saving} className="px-3 py-2 rounded-lg text-sm bg-text text-bg disabled:opacity-40">{t('save')}</button>
                      <button type="button" onClick={() => setEditing(false)} className="px-3 py-2 text-sm text-text-2">{t('cancel')}</button>
                    </form>
                  ) : (
                    <h2 className="text-[24px] font-semibold tracking-[-0.02em] flex-1">{activeList.name}</h2>
                  )}
                  {!editing && (
                    <>
                      <button
                        type="button"
                        onClick={() => { setRenameValue(activeList.name); setEditing(true); }}
                        disabled={saving}
                        className="inline-flex items-center gap-1.5 text-sm text-text-2 hover:text-text px-2 py-1.5"
                      >
                        <Pencil className="w-3.5 h-3.5" />{t('renameList')}
                      </button>
                      <button
                        type="button"
                        onClick={() => void deleteList()}
                        disabled={saving}
                        className="inline-flex items-center gap-1.5 text-sm text-hard px-2 py-1.5 disabled:opacity-40"
                      >
                        <Trash2 className="w-3.5 h-3.5" />{t('deleteList')}
                      </button>
                    </>
                  )}
                </div>
                <p className="text-xs mono text-text-3 mb-4">
                  {t('listProblems', { n: activeList.problemIds.length })} · {t('solvedCount', { solved: solvedCount, total: activeList.problemIds.length })}
                </p>

                <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--line)' }}>
                  {listedProblems.length > 0 ? (
                    <div className="divide-y divide-[var(--line)]">
                      {listedProblems.map((problem) => {
                        const status = progress[problem.id]?.status ?? 'todo';
                        return (
                          <div key={problem.id} className="flex items-center gap-3 py-3 px-4 bg-bg-elev hover:bg-[color-mix(in_oklab,var(--accent)_3%,var(--bg-elev))]">
                            <StatusIcon status={status} />
                            <Link href={'/problems/' + problem.id} className="flex-1 min-w-0">
                              <div className="truncate text-sm font-medium hover:text-accent">
                                {locale === 'zh' ? problem.titleZh : problem.title}
                              </div>
                              <div className="mono text-[11px] text-text-3 mt-0.5">{problem.id}</div>
                            </Link>
                            <Badge variant={problem.difficulty.toLowerCase() as 'easy' | 'medium' | 'hard'}>
                              {problem.difficulty.toUpperCase()}
                            </Badge>
                            <StudyListPickerButton
                              compact
                              problemId={problem.id}
                              problemTitle={locale === 'zh' ? problem.titleZh : problem.title}
                            />
                            <button
                              type="button"
                              title={t('deleteList') + ' / ' + t('addToList')}
                              aria-label={t('deleteList') + ': ' + problem.id}
                              disabled={saving}
                              onClick={() => void mutate({
                                action: 'remove', listId: activeList.id, taskId: problem.id,
                              })}
                              className="w-8 h-8 flex items-center justify-center rounded-lg text-text-3 hover:text-hard disabled:opacity-40"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="px-6 py-16 text-center">
                      <Bookmark className="w-7 h-7 text-text-3 mx-auto mb-3" />
                      <p className="text-sm text-text-2">{t('emptyStudyList')}</p>
                      <Link href="/problems" className="inline-block mt-4 text-sm text-accent hover:underline">
                        {t('problems')}
                      </Link>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="rounded-xl p-12 text-center text-text-3 text-sm" style={{ border: '1px dashed var(--line)' }}>
                {loading ? t('loading') : t('noStudyLists')}
                <div className="mt-3">{t('myListsSubtitle')}</div>
              </div>
            )}
            {catalogError && <p role="alert" className="mt-4 text-sm text-hard">{catalogError}</p>}
          </section>
        </div>
      </main>
      <Footer />
    </div>
  );
}
