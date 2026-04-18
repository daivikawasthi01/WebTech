import { lazy, Suspense, useState, useEffect, useCallback } from 'react'
import { api } from './utils/api.js'

import PipelineTab    from './components/PipelineTab.jsx'

import SideNavBar     from './components/layout/SideNavBar.jsx'
import TopAppBar      from './components/layout/TopAppBar.jsx'
import SettingsDrawer from './components/layout/SettingsDrawer.jsx'

const GAResultsTab = lazy(() => import('./components/GAResultsTab.jsx'))
const FileRiskTab = lazy(() => import('./components/FileRiskTab.jsx'))
const SHAPTab = lazy(() => import('./components/SHAPTab.jsx'))
const AdvancedTab = lazy(() => import('./components/AdvancedTab.jsx'))
const ChromosomeEditor = lazy(() => import('./components/ChromosomeEditor.jsx'))
const BaselinesTab = lazy(() =>
  import('./components/OtherTabs.jsx').then(mod => ({ default: mod.BaselinesTab }))
)
const AblationTab = lazy(() =>
  import('./components/OtherTabs.jsx').then(mod => ({ default: mod.AblationTab }))
)
const MultiRepoTab = lazy(() =>
  import('./components/OtherTabs.jsx').then(mod => ({ default: mod.MultiRepoTab }))
)
const SensitivityTab = lazy(() =>
  import('./components/OtherTabs.jsx').then(mod => ({ default: mod.SensitivityTab }))
)

const TABS_INFO = {
  'pipeline':    'Pipeline',
  'ga':          'GA Results',
  'baselines':   'Baselines',
  'ablation':    'Ablation',
  'filerisk':    'File Risk',
  'shap':        'SHAP Explanations',
  'chromeditor': 'Chromosome Editor',
  'advanced':    'Advanced Settings',
  'multirepo':   'Multi-Repo Stats',
  'sensitivity': 'Sensitivity Sweep',
};

const DEFAULT_CFG = {
  repo_path:      'test_repos/flask',
  raw_file:       'data/flask_dataset.csv',
  processed_file: 'data/flask_dataset_clean.csv',
  pop_size:       15,
  generations:    10,
  alpha:          1.0,
  beta:           0.5,
  mutation_rate:  0.20,
  min_mutation:   0.03,
  stagnation:     5,
  n_trials:       20,
  tune_trials:    50,
  repos:          ['flask', 'requests', 'django'],
  run_tuning:     false,
  run_baselines:  false,
  run_ablation:   false,
  run_stats:      false,
  run_multi:      false,
  run_sensitivity:false,
  run_report:     true,
  run_all:        false,
  force_collect:  false,
  force_process:  false,
}

export default function App() {
  const [activeTab, setActiveTab] = useState('pipeline')
  const [cfg, setCfg]             = useState(DEFAULT_CFG)
  const [fileStatus, setFileStatus] = useState({})
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [liveGAActive, setLiveGAActive] = useState(false)

  // Poll file status every 3s so sidebar indicators update automatically
  const refreshStatus = useCallback(() => {
    api.status().then(r => setFileStatus(r.data.files)).catch(() => {})
  }, [])

  useEffect(() => {
    refreshStatus()
    const t = setInterval(refreshStatus, 3000)
    return () => clearInterval(t)
  }, [refreshStatus])

  const TabContent = {
    pipeline: (
      <PipelineTab
        cfg={cfg}
        fileStatus={fileStatus}
        onRefresh={refreshStatus}
        onJobStart={() => setLiveGAActive(true)}
        onJobDone={() => setLiveGAActive(false)}
      />
    ),
    ga: <GAResultsTab cfg={cfg} liveActive={liveGAActive} />,
    baselines: <BaselinesTab />,
    ablation: <AblationTab />,
    filerisk: <FileRiskTab cfg={cfg} />,
    shap: <SHAPTab cfg={cfg} />,
    chromeditor: <ChromosomeEditor cfg={cfg} />,
    advanced: <AdvancedTab />,
    multirepo: <MultiRepoTab />,
    sensitivity: <SensitivityTab />,
  }

  return (
    <div className="min-h-screen bg-background text-on-surface font-sans overflow-hidden">
      <SideNavBar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        toggleSettings={() => setIsSettingsOpen(true)} 
      />

      <div className="flex-1 flex flex-col min-h-screen relative">
        <TopAppBar 
          activeTabLabel={TABS_INFO[activeTab]} 
          systemActive={liveGAActive} 
        />

        <main className="flex-1 ml-64 p-8 overflow-y-auto" style={{ height: 'calc(100vh - 64px)' }}>
          <Suspense
            fallback={
              <div className="glass-panel rounded-xl min-h-[320px] flex items-center justify-center border border-outline-variant">
                <p className="text-sm tracking-widest text-on-surface-variant uppercase">
                  Loading module...
                </p>
              </div>
            }
          >
            {TabContent[activeTab]}
          </Suspense>
        </main>
      </div>

      <SettingsDrawer 
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        cfg={cfg}
        setCfg={setCfg}
        fileStatus={fileStatus}
      />
    </div>
  )
}
