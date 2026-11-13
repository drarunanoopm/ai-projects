import type { WorkspaceTab } from '../types/bioacoustics';

interface TopBarProps {
  activeTab: WorkspaceTab;
  onSelectTab: (tab: WorkspaceTab) => void;
  onTriggerUpload: () => void;
}

const NAV_ITEMS: Array<{ id: WorkspaceTab; label: string }> = [
  { id: 'lab', label: 'Spectrogram Lab' },
  { id: 'architecture', label: 'CNN Architecture' },
  { id: 'dataset', label: 'Dataset & Training' },
  { id: 'herd', label: 'Herd Log' },
  { id: 'about', label: 'About Me' },
];

export function TopBar({ activeTab, onSelectTab, onTriggerUpload }: TopBarProps) {
  return (
    <header className="sticky top-0 z-30 bg-white border-b border-slate-200 px-6 h-14 flex items-center justify-between gap-8">
      <button
        type="button"
        onClick={() => onSelectTab('lab')}
        className="font-display text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-600 cursor-pointer"
      >
        BovAcoustic
      </button>

      <nav className="flex items-center gap-6 overflow-x-auto" aria-label="Primary sections">
        {NAV_ITEMS.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectTab(item.id)}
              className={`relative py-4 text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-600 ${
                isActive ? 'text-sky-700' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {item.label}
              {isActive && (
                <span
                  aria-hidden="true"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-sky-600"
                />
              )}
            </button>
          );
        })}
      </nav>

      <div className="shrink-0">
        <button
          type="button"
          onClick={onTriggerUpload}
          className="px-4 py-2 text-xs font-medium text-white bg-sky-700 hover:bg-sky-800 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-600"
        >
          Upload Audio
        </button>
      </div>
    </header>
  );
}
