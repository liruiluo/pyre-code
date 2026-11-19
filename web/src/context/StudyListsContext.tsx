'use client';

import * as Dialog from '@radix-ui/react-dialog';
import {
  createContext, useCallback, useContext, useEffect, useMemo,
  useRef, useState, type FormEvent, type ReactNode,
} from 'react';
import { Check, Plus, X } from 'lucide-react';
import { useLocale } from '@/context/LocaleContext';
import type { StudyList } from '@/lib/types';

export type StudyListAction =
  | { action: 'create'; name: string }
  | { action: 'rename'; listId: number; name: string }
  | { action: 'delete'; listId: number }
  | { action: 'add' | 'remove'; listId: number; taskId: string };

type PickerTarget = { id: string; title: string };

interface StudyListsValue {
  lists: StudyList[];
  loading: boolean;
  saving: boolean;
  error: string | null;
  clearError: () => void;
  openPicker: (target: PickerTarget) => void;
  mutate: (action: StudyListAction) => Promise<StudyList[] | null>;
}

const StudyListsContext = createContext<StudyListsValue | null>(null);
const API_URL = (process.env.NEXT_PUBLIC_BASE_PATH ?? '') + '/api/study-lists';

export function StudyListsProvider({ children }: { children: ReactNode }) {
  const [lists, setLists] = useState<StudyList[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [picker, setPicker] = useState<PickerTarget | null>(null);
  const busy = useRef(false);

  useEffect(() => {
    let mounted = true;
    fetch(API_URL, { cache: 'no-store' })
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Failed to load study lists');
        return data as { lists: StudyList[] };
      })
      .then((data) => { if (mounted) setLists(data.lists); })
      .catch((err) => { if (mounted) setError(String(err.message || err)); })
      .finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; };
  }, []);

  const mutate = useCallback(async (action: StudyListAction): Promise<StudyList[] | null> => {
    if (busy.current) return null;
    busy.current = true;
    setSaving(true);
    setError(null);
    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(action),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Failed to save study list');
      const updated = (data as { lists: StudyList[] }).lists;
      setLists(updated);
      return updated;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save study list');
      return null;
    } finally {
      busy.current = false;
      setSaving(false);
    }
  }, []);

  const value = useMemo<StudyListsValue>(() => ({
    lists, loading, saving, error,
    clearError: () => setError(null),
    openPicker: (target) => { setError(null); setPicker(target); },
    mutate,
  }), [lists, loading, saving, error, mutate]);

  return (
    <StudyListsContext.Provider value={value}>
      {children}
      <StudyListPickerDialog target={picker} onClose={() => setPicker(null)} />
    </StudyListsContext.Provider>
  );
}

export function useStudyLists(): StudyListsValue {
  const value = useContext(StudyListsContext);
  if (!value) throw new Error('StudyListsProvider is missing');
  return value;
}

function StudyListPickerDialog({ target, onClose }: { target: PickerTarget | null; onClose: () => void }) {
  const { t } = useLocale();
  const { lists, loading, saving, error, mutate } = useStudyLists();
  const [newName, setNewName] = useState('');

  useEffect(() => { setNewName(''); }, [target?.id]);

  const createAndAdd = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = newName.trim();
    if (!target || !name || saving) return;
    const updated = await mutate({ action: 'create', name });
    const created = updated?.find((list) => list.name === name);
    if (created) {
      setNewName('');
      await mutate({ action: 'add', listId: created.id, taskId: target.id });
    }
  };

  return (
    <Dialog.Root open={target !== null} onOpenChange={(isOpen) => { if (!isOpen) onClose(); }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[100] bg-black/45" />
        <Dialog.Content
          className="fixed left-1/2 top-1/2 z-[101] w-[min(92vw,440px)] -translate-x-1/2 -translate-y-1/2 rounded-2xl p-6 shadow-2xl outline-none"
          style={{ background: 'var(--bg-elev)', border: '1px solid var(--line)' }}
        >
          <div className="flex items-start gap-3 mb-1">
            <Dialog.Title className="flex-1 text-xl font-semibold">{t('organizeProblem')}</Dialog.Title>
            <Dialog.Close className="p-1 rounded-md text-text-3 hover:text-text" aria-label={t('close')}>
              <X className="w-4 h-4" />
            </Dialog.Close>
          </div>
          <Dialog.Description className="text-sm text-text-2 mb-4">
            {target?.title} · {t('multipleListsHint')}
          </Dialog.Description>

          <div className="flex flex-col gap-1 max-h-64 overflow-y-auto mb-5">
            {loading && <p className="py-4 text-sm text-text-3">{t('loading')}</p>}
            {!loading && lists.length === 0 && (
              <p className="py-4 text-sm text-text-3">{t('noStudyLists')}</p>
            )}
            {lists.map((list) => {
              const checked = !!target && list.problemIds.includes(target.id);
              return (
                <button
                  key={list.id}
                  type="button"
                  role="checkbox"
                  aria-checked={checked}
                  disabled={saving}
                  onClick={() => {
                    if (target) void mutate({
                      action: checked ? 'remove' : 'add',
                      listId: list.id,
                      taskId: target.id,
                    });
                  }}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left text-sm hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)] disabled:opacity-50"
                >
                  <span
                    className="w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0"
                    style={{
                      border: checked ? '1px solid var(--accent)' : '1px solid var(--line-strong)',
                      background: checked ? 'var(--accent)' : 'var(--bg-sunken)',
                      color: checked ? 'var(--bg)' : 'var(--text)',
                    }}
                  >
                    {checked && <Check className="w-3.5 h-3.5" />}
                  </span>
                  <span className="flex-1 truncate">{list.name}</span>
                  <span className="mono text-xs text-text-3 tabular-nums">{list.problemIds.length}</span>
                </button>
              );
            })}
          </div>

          <form onSubmit={(event) => void createAndAdd(event)} className="flex gap-2">
            <input
              type="text"
              maxLength={40}
              value={newName}
              onChange={(event) => setNewName(event.target.value)}
              placeholder={t('listNamePlaceholder')}
              aria-label={t('listNamePlaceholder')}
              className="flex-1 min-w-0 h-9 px-3 rounded-lg text-sm bg-bg text-text outline-none focus:border-accent"
              style={{ border: '1px solid var(--line)' }}
            />
            <button
              type="submit"
              disabled={!newName.trim() || saving || loading}
              className="h-9 px-3 rounded-lg inline-flex items-center gap-1 text-sm font-medium bg-text text-bg disabled:opacity-40"
            >
              <Plus className="w-3.5 h-3.5" />{t('createList')}
            </button>
          </form>
          {error && <p className="mt-3 text-sm text-hard" role="alert">{error}</p>}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
