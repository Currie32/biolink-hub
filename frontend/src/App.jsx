import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { stemmer } from 'stemmer'
import './App.css'

// ─── Constants ───────────────────────────────────────────────────────────────

const ENTITY_TYPES = ['GENE', 'DISEASE', 'CHEMICAL', 'VARIANT', 'ORGANISM', 'CELL_TYPE']
const REL_TYPES = [
  'upregulates', 'downregulates', 'inhibits', 'binds',
  'increases_risk', 'associated_with', 'regulates', 'expressed_in',
]

// ─── Routing ─────────────────────────────────────────────────────────────────

function parseRoute() {
  const path = window.location.pathname
  if (path === '/annotate') return { view: 'annotate-dashboard' }
  if (path.startsWith('/annotate/')) {
    const id = parseInt(path.slice('/annotate/'.length), 10)
    if (!isNaN(id)) return { view: 'annotate-view', id }
  }
  if (path.startsWith('/type/')) return { view: 'type', type: decodeURIComponent(path.slice(6)) }
  if (path.startsWith('/entity/')) return { view: 'entity', id: decodeURIComponent(path.slice(8)) }
  return { view: 'home' }
}

function navigate(path) {
  window.history.pushState(null, '', path)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Stem each whitespace-separated token and rejoin.
function stemPhrase(text) {
  return text.toLowerCase().trim().split(/\s+/).map(stemmer).join(' ')
}

// Two entity spans are considered similar if their stemmed forms are identical.
// Examples: "mentally retarded" ↔ "mental retardation" (both → "mental retard"),
//           "patient" ↔ "patients" (both → "patient").
function areSimilarTexts(a, b) {
  if (a.toLowerCase().trim() === b.toLowerCase().trim()) return true
  return stemPhrase(a) === stemPhrase(b)
}

// Auto-merge entities of the same type that look like morphological variants.
// Earlier occurrences (lower start offset) are kept as the canonical form.
function autoMergeEntities(entities) {
  const result = entities.map(e => ({ ...e }))
  for (let i = 0; i < result.length; i++) {
    if (result[i].status === 'rejected' || result[i].status === 'merged') continue
    for (let j = i + 1; j < result.length; j++) {
      if (result[j].status === 'rejected' || result[j].status === 'merged') continue
      if (result[i].type !== result[j].type) continue
      if (areSimilarTexts(result[i].text, result[j].text)) {
        result[j] = { ...result[j], status: 'merged', mergedInto: result[i].text }
      }
    }
  }
  return result
}

function findAllOccurrences(text, query) {
  const lower = text.toLowerCase()
  const q = query.toLowerCase()
  const matches = []
  let idx = 0
  while ((idx = lower.indexOf(q, idx)) !== -1) {
    matches.push({ start: idx, end: idx + q.length })
    idx += q.length
  }
  return matches
}

function inSameSentence(text, e1, e2) {
  const lo = Math.min(e1.end, e2.end)
  const hi = Math.max(e1.start, e2.start)
  if (lo >= hi) return true
  return !/[.!?]/.test(text.slice(lo, hi))
}

function findSentenceContext(text, e1, e2) {
  const lo = Math.min(e1.start, e2.start)
  const hi = Math.max(e1.end, e2.end)
  let start = 0
  for (let i = lo - 1; i >= 0; i--) {
    if (/[.!?]/.test(text[i])) { start = i + 1; break }
  }
  while (start < lo && /\s/.test(text[start])) start++
  let end = text.length
  for (let i = hi; i < text.length; i++) {
    if (/[.!?]/.test(text[i])) { end = i + 1; break }
  }
  return { context: text.slice(start, end).trim(), contextStart: start }
}

// ─── Main App ────────────────────────────────────────────────────────────────

function App() {
  const [hubView, setHubView] = useState(() => parseRoute())
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [entity, setEntity] = useState(null)
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)
  const [typeFilter, setTypeFilter] = useState(null)
  const [typeEntities, setTypeEntities] = useState([])

  useEffect(() => {
    const handleRoute = () => {
      const route = parseRoute()
      setHubView(route)
      if (route.view === 'annotate-dashboard' || route.view === 'annotate-view') return
      if (route.view === 'type') {
        setEntity(null); setQuery(''); setShowDropdown(false); setTypeFilter(route.type)
        setLoading(true)
        fetch(`/api/entities?type=${encodeURIComponent(route.type)}`)
          .then(r => r.json())
          .then(data => { setTypeEntities(data); setLoading(false) })
          .catch(() => setLoading(false))
      } else if (route.view === 'entity') {
        setShowDropdown(false); setTypeFilter(null); setTypeEntities([])
        setLoading(true)
        fetch(`/api/entity/${encodeURIComponent(route.id)}`)
          .then(r => r.json())
          .then(data => { setEntity(data); setLoading(false) })
          .catch(() => setLoading(false))
      } else {
        setEntity(null); setTypeFilter(null); setTypeEntities([])
      }
    }
    handleRoute()
    window.addEventListener('popstate', handleRoute)
    return () => window.removeEventListener('popstate', handleRoute)
  }, [])

  useEffect(() => {
    fetch('/api/stats').then(r => r.json()).then(setStats).catch(() => {})
  }, [])

  useEffect(() => {
    if (!query.trim()) { setResults([]); setShowDropdown(false); return }
    const timer = setTimeout(() => {
      fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then(r => r.json())
        .then(data => {
          const seen = new Map()
          for (const r of data) {
            const key = `${r.name.toLowerCase()}::${r.type}`
            const ex = seen.get(key)
            if (!ex || (r.description && !ex.description)) seen.set(key, r)
          }
          setResults([...seen.values()])
          setShowDropdown(true)
        })
        .catch(() => {})
    }, 150)
    return () => clearTimeout(timer)
  }, [query])

  const selectEntity = useCallback((id) => { setShowDropdown(false); navigate(`/entity/${encodeURIComponent(id)}`) }, [])
  const selectType = useCallback((type) => { setQuery(''); navigate(`/type/${encodeURIComponent(type)}`) }, [])
  const goHome = useCallback(() => { setQuery(''); navigate('/') }, [])

  if (hubView.view === 'annotate-dashboard') return <AnnotationDashboard />
  if (hubView.view === 'annotate-view') return <AnnotationView id={hubView.id} />

  return (
    <div className="app">
      <header className="header">
        <h1 onClick={goHome} style={{ cursor: 'pointer' }}>BioLink Hub</h1>
        <p>Open Biomedical Knowledge Explorer</p>
        <nav className="header-nav">
          <a className="header-nav-link" onClick={() => navigate('/annotate')}>Annotation Tool</a>
        </nav>
      </header>

      <SearchBar
        query={query} setQuery={setQuery} results={results}
        showDropdown={showDropdown} setShowDropdown={setShowDropdown} onSelect={selectEntity}
      />

      {!entity && !loading && !typeFilter && stats && !query && (
        <StatsBar stats={stats} onTypeClick={selectType} />
      )}
      {loading && <div className="loading">Loading...</div>}
      {!entity && !loading && typeFilter && (
        <EntityList type={typeFilter} entities={typeEntities} onSelect={selectEntity} onBack={goHome} />
      )}
      {entity && !loading && (
        <EntityDetail entity={entity} onNavigate={selectEntity} />
      )}
    </div>
  )
}

// ─── Hub Components ───────────────────────────────────────────────────────────

function SearchBar({ query, setQuery, results, showDropdown, setShowDropdown, onSelect }) {
  const containerRef = useRef(null)
  const inputRef = useRef(null)
  const [highlightIdx, setHighlightIdx] = useState(-1)

  useEffect(() => {
    const handleClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) setShowDropdown(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [setShowDropdown])

  useEffect(() => { setHighlightIdx(-1) }, [results])

  const handleKeyDown = (e) => {
    if (!showDropdown || results.length === 0) return
    if (e.key === 'ArrowDown') { e.preventDefault(); setHighlightIdx(i => Math.min(i + 1, results.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setHighlightIdx(i => Math.max(i - 1, 0)) }
    else if (e.key === 'Enter' && highlightIdx >= 0) { e.preventDefault(); onSelect(results[highlightIdx].id) }
    else if (e.key === 'Escape') setShowDropdown(false)
  }

  return (
    <div className="search-container" ref={containerRef}>
      <input
        ref={inputRef} className="search-input" type="text"
        placeholder="Search genes, diseases, drugs, pathways..."
        value={query}
        onChange={e => { setQuery(e.target.value); setShowDropdown(true) }}
        onFocus={() => { if (results.length > 0) setShowDropdown(true) }}
        onKeyDown={handleKeyDown} autoFocus
      />
      {showDropdown && results.length > 0 && (
        <ul className="autocomplete-dropdown">
          {results.slice(0, 12).map((r, i) => (
            <li
              key={r.id}
              className={`autocomplete-item ${i === highlightIdx ? 'highlighted' : ''}`}
              onMouseDown={() => onSelect(r.id)}
              onMouseEnter={() => setHighlightIdx(i)}
            >
              <span className="ac-name">{r.name}</span>
              <span className={`type-badge type-${r.type}`}>{formatType(r.type)}</span>
              {r.description && (
                <span className="ac-desc">{r.description.slice(0, 80)}{r.description.length > 80 ? '...' : ''}</span>
              )}
            </li>
          ))}
        </ul>
      )}
      {!query && <p className="search-hint">Try: SNCA, APOE, Alzheimer, neuroinflammation, dopamine</p>}
    </div>
  )
}

function StatsBar({ stats, onTypeClick }) {
  const sorted = Object.entries(stats.entities || {}).sort((a, b) => b[1] - a[1])
  return (
    <div className="stats-section">
      <div className="stats-grid">
        {sorted.map(([type, count]) => (
          <div key={type} className="stat-chip stat-chip-clickable" onClick={() => onTypeClick(type)}>
            <span className="stat-count">{count}</span>
            <span className="stat-label">{formatType(type)}s</span>
          </div>
        ))}
      </div>
      <div className="stats-summary">
        {stats.total_entities} entities
        {stats.total_relationships > 0 && <> &middot; {stats.total_relationships} relationships</>}
        {stats.total_papers > 0 && <> &middot; {stats.total_papers} papers</>}
      </div>
    </div>
  )
}

function EntityList({ type, entities, onSelect, onBack }) {
  return (
    <div className="entity-list-view">
      <div className="list-header">
        <button className="back-link" onClick={onBack}>&larr; All types</button>
        <h2>
          <span className={`type-badge type-${type}`}>{formatType(type)}</span>
          <span className="list-count">{entities.length}</span>
        </h2>
      </div>
      <div className="entity-list">
        {entities.map(e => (
          <div key={e.id} className="entity-list-item" onClick={() => onSelect(e.id)}>
            <div className="entity-list-name">{e.name}</div>
            {e.description && (
              <div className="entity-list-desc">
                {e.description.length > 120 ? e.description.slice(0, 120) + '...' : e.description}
              </div>
            )}
            <div className="entity-list-id">{e.id}</div>
          </div>
        ))}
        {entities.length === 0 && <div className="empty-state">No entities found.</div>}
      </div>
    </div>
  )
}

function EntityDetail({ entity, onNavigate }) {
  const synonyms = entity.synonyms || []
  const externalIds = entity.external_ids || {}
  const metadata = entity.metadata || {}
  const relationships = entity.relationships || []
  const [expandedRel, setExpandedRel] = useState(null)
  const [evidence, setEvidence] = useState({})

  const toggleEvidence = (relId) => {
    if (expandedRel === relId) { setExpandedRel(null); return }
    setExpandedRel(relId)
    if (!evidence[relId]) {
      fetch(`/api/relationship/${relId}/evidence`)
        .then(r => r.json())
        .then(data => setEvidence(prev => ({ ...prev, [relId]: data })))
        .catch(() => {})
    }
  }

  const groupedRels = useMemo(() => {
    const groups = {}
    for (const r of relationships) {
      if (!groups[r.type]) groups[r.type] = []
      groups[r.type].push(r)
    }
    return Object.entries(groups).sort((a, b) => b[1].length - a[1].length)
  }, [relationships])

  const metaEntries = Object.entries(metadata).filter(([, v]) => v)

  return (
    <div className="entity-detail">
      <div className="detail-header">
        <div className="detail-title-row">
          <h2>{entity.name}</h2>
          <span className={`type-badge type-${entity.type}`}>{formatType(entity.type)}</span>
        </div>
        <div className="detail-id">{entity.id}</div>
      </div>

      <div className="info-grid">
        <div className="info-main">
          {entity.description && (
            <div className="detail-section"><h3>Description</h3><p>{entity.description}</p></div>
          )}
          {metaEntries.length > 0 && (
            <div className="detail-section">
              <h3>Details</h3>
              <dl className="metadata-grid">
                {metaEntries.map(([key, val]) => (
                  <div key={key} className="meta-row">
                    <dt>{key.replace(/_/g, ' ')}</dt>
                    <dd>{String(val)}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
          {synonyms.length > 0 && (
            <div className="detail-section">
              <h3>Synonyms</h3>
              <div className="synonyms-list">
                {synonyms.map((s, i) => <span key={i} className="synonym-chip">{s}</span>)}
              </div>
            </div>
          )}
          {Object.keys(externalIds).length > 0 && (
            <div className="detail-section">
              <h3>External IDs</h3>
              <div className="ext-ids-inline">
                {Object.entries(externalIds).map(([db, id]) => (
                  <span key={db} className="ext-id-chip"><strong>{db}:</strong> {id}</span>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="info-sidebar">
          <div className="sidebar-stat">
            <div className="sidebar-stat-value">{relationships.length}</div>
            <div className="sidebar-stat-label">Relationships</div>
          </div>
          {entity.paper_count > 0 && (
            <div className="sidebar-stat">
              <div className="sidebar-stat-value">{entity.paper_count}</div>
              <div className="sidebar-stat-label">Papers</div>
            </div>
          )}
          {groupedRels.length > 0 && (
            <div className="sidebar-types">
              {groupedRels.map(([type, rels]) => (
                <div key={type} className="sidebar-type-row">
                  <span className="rel-type-badge">{type.replace(/_/g, ' ')}</span>
                  <span className="sidebar-type-count">{rels.length}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {groupedRels.length > 0 && (
        <div className="detail-section relationships-section">
          <h3>Relationships</h3>
          {groupedRels.map(([type, rels]) => (
            <div key={type} className="rel-group">
              <div className="rel-group-header">
                <span className="rel-type-badge">{type.replace(/_/g, ' ')}</span>
                <span className="rel-group-count">{rels.length}</span>
              </div>
              <div className="rel-group-items">
                {rels.map((r, i) => (
                  <div key={i}>
                    <div className="rel-item">
                      <a className="rel-entity-link" onClick={() => onNavigate(r.target_id === entity.id ? r.source_id : r.target_id)}>
                        {r.related_name}
                      </a>
                      {r.confidence != null && (
                        <span className={`confidence-badge confidence-${confidenceTier(r.confidence)}`}>
                          {Math.round(r.confidence * 100)}%
                        </span>
                      )}
                      {r.evidence_count > 0 && (
                        <button className="evidence-toggle" onClick={(e) => { e.stopPropagation(); toggleEvidence(r.id) }}>
                          {r.evidence_count} paper{r.evidence_count !== 1 ? 's' : ''} {expandedRel === r.id ? '\u25B4' : '\u25BE'}
                        </button>
                      )}
                    </div>
                    {expandedRel === r.id && evidence[r.id] && (
                      <div className="evidence-list">
                        {evidence[r.id].map((ev, j) => (
                          <div key={j} className="evidence-item">
                            <a className="evidence-title" href={ev.source_url} target="_blank" rel="noopener noreferrer">{ev.paper_title}</a>
                            <span className="evidence-meta">{ev.journal} ({ev.year})</span>
                            <div className="evidence-tags">
                              {ev.organism && <span className="evidence-tag">{ev.organism}</span>}
                              {ev.cell_type && <span className="evidence-tag">{ev.cell_type}</span>}
                              {ev.experiment_type && <span className="evidence-tag">{ev.experiment_type}</span>}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {relationships.length === 0 && (
        <div className="detail-section">
          <h3>Relationships</h3>
          <p className="muted">No connections found yet.</p>
        </div>
      )}
    </div>
  )
}

// ─── Annotation Dashboard ─────────────────────────────────────────────────────

function AnnotationDashboard() {
  const [queue, setQueue] = useState([])
  const [loading, setLoading] = useState(true)
  const [annotatorName, setAnnotatorName] = useState(() => localStorage.getItem('annotator_name') || '')
  const [nameInput, setNameInput] = useState('')
  const [iaa, setIaa] = useState({})

  useEffect(() => {
    fetch('/api/annotate/queue')
      .then(r => r.json())
      .then(data => { setQueue(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  useEffect(() => {
    queue.filter(a => a.completed_count >= 2).forEach(a => {
      fetch(`/api/annotate/iaa/${a.id}`)
        .then(r => r.json())
        .then(data => setIaa(prev => ({ ...prev, [a.id]: data })))
        .catch(() => {})
    })
  }, [queue])

  const [fetchKeywords, setFetchKeywords] = useState('')
  const [fetchCount, setFetchCount] = useState(10)
  const [fetching, setFetching] = useState(false)
  const [fetchMsg, setFetchMsg] = useState('')

  const saveName = () => {
    const name = nameInput.trim()
    if (!name) return
    localStorage.setItem('annotator_name', name)
    setAnnotatorName(name)
  }

  const fetchAbstracts = async () => {
    if (!fetchKeywords.trim()) return
    setFetching(true)
    setFetchMsg('')
    try {
      const res = await fetch('/api/annotate/fetch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keywords: fetchKeywords, max_results: fetchCount }),
      })
      const data = await res.json()
      setFetchMsg(data.added > 0 ? `Added ${data.added} abstract${data.added !== 1 ? 's' : ''}` : 'No new abstracts found')
      if (data.added > 0) {
        fetch('/api/annotate/queue').then(r => r.json()).then(setQueue).catch(() => {})
      }
    } catch {
      setFetchMsg('Fetch failed')
    } finally {
      setFetching(false)
    }
  }

  if (!annotatorName) {
    return (
      <div className="app annotator-prompt-page">
        <div className="annotator-prompt">
          <h2>Welcome to the Annotation Tool</h2>
          <p>Enter your name to get started. This will be saved for future sessions.</p>
          <div className="prompt-row">
            <input
              className="search-input" placeholder="Your name..."
              value={nameInput} onChange={e => setNameInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && saveName()} autoFocus
            />
            <button className="btn-primary" onClick={saveName}>Start</button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-row">
          <h1>Annotation Tool</h1>
          <button className="back-link" onClick={() => navigate('/')}>&larr; BioLink Hub</button>
        </div>
        <p>
          Annotating as <strong>{annotatorName}</strong> &middot;{' '}
          <a className="change-link" onClick={() => { localStorage.removeItem('annotator_name'); setAnnotatorName('') }}>
            change
          </a>
        </p>
      </header>

      {loading && <div className="loading">Loading queue...</div>}

      {!loading && (
        <div className="queue-section">
          <div className="queue-header">
            <h2>Annotation Queue ({queue.length} abstract{queue.length !== 1 ? 's' : ''})</h2>
            <a href="/api/annotate/export" className="export-link" target="_blank" rel="noopener noreferrer">
              Export JSONL
            </a>
          </div>
          <div className="fetch-bar">
            <input
              className="search-input fetch-input"
              placeholder="Keywords to fetch from PubMed (e.g. TREM2 Alzheimer disease)..."
              value={fetchKeywords}
              onChange={e => setFetchKeywords(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && fetchAbstracts()}
            />
            <input
              type="number" className="fetch-count-input" min={1} max={50}
              value={fetchCount} onChange={e => setFetchCount(parseInt(e.target.value) || 10)}
              title="Max results"
            />
            <button className="btn-primary btn-small" onClick={fetchAbstracts} disabled={fetching}>
              {fetching ? 'Fetching...' : 'Fetch'}
            </button>
            {fetchMsg && <span className="fetch-msg">{fetchMsg}</span>}
          </div>
          <div className="queue-table-wrap">
            <table className="queue-table">
              <thead>
                <tr>
                  <th>PMID</th><th>Source</th><th>Annotators</th><th>Your status</th><th>IAA</th><th></th>
                </tr>
              </thead>
              <tbody>
                {queue.map(item => {
                  const mine = item.annotators.find(a => a.name === annotatorName)
                  const statusLabel = mine
                    ? (mine.phase === 'completed' ? 'Done' : `In progress`)
                    : 'Not started'
                  const iaaData = iaa[item.id]
                  return (
                    <tr key={item.id} className="queue-row" onClick={() => navigate(`/annotate/${item.id}`)}>
                      <td className="queue-pmid">{item.pmid || '—'}</td>
                      <td><span className="source-badge">{item.source}</span></td>
                      <td className="queue-annotators">
                        {item.annotators.length === 0 ? '—' : item.annotators.map(a => a.name).join(', ')}
                      </td>
                      <td>
                        <span className={`status-badge status-${mine ? mine.phase : 'none'}`}>
                          {statusLabel}
                        </span>
                      </td>
                      <td className="queue-iaa">
                        {iaaData?.entity_agreement != null
                          ? `${Math.round(iaaData.entity_agreement * 100)}%`
                          : '—'}
                      </td>
                      <td>
                        <button
                          className="btn-small btn-ghost"
                          onClick={e => { e.stopPropagation(); navigate(`/annotate/${item.id}`) }}
                        >
                          {mine ? 'Continue' : 'Start'}
                        </button>
                      </td>
                    </tr>
                  )
                })}
                {queue.length === 0 && (
                  <tr>
                    <td colSpan={6} className="empty-state">
                      No abstracts in queue. Run populate_annotation_queue.py first.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Annotation View ──────────────────────────────────────────────────────────

function AnnotationView({ id }) {
  const [abstract, setAbstract] = useState(null)
  const [loading, setLoading] = useState(true)
  const [annotatorName] = useState(() => localStorage.getItem('annotator_name') || 'anonymous')
  const [entities, setEntities] = useState([])
  const [relationships, setRelationships] = useState([])
  const [pendingSelection, setPendingSelection] = useState(null)
  const [editingEntity, setEditingEntity] = useState(null)
  const [newType, setNewType] = useState('GENE')
  const [saving, setSaving] = useState(false)
  const [savedMsg, setSavedMsg] = useState('')
  const [hoveredPair, setHoveredPair] = useState(null)

  const entityRef = useRef([])
  const relRef = useRef([])
  useEffect(() => { entityRef.current = entities }, [entities])
  useEffect(() => { relRef.current = relationships }, [relationships])

  // Load abstract + existing annotation
  useEffect(() => {
    fetch(`/api/annotate/${id}?annotator_name=${encodeURIComponent(annotatorName)}`)
      .then(r => r.json())
      .then(data => {
        setAbstract(data.abstract)
        if (data.existing_annotation) {
          const existing = data.existing_annotation
          relRef.current = existing.relationships  // set before entities triggers recompute
          setEntities(existing.entities)
          setRelationships(existing.relationships)
        } else {
          const pre = data.abstract.prelabels
          const initial = (pre.entities || []).map(e => ({ ...e, status: 'accepted' }))
          setEntities(autoMergeEntities(initial))
        }
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [id, annotatorName])

  // Recompute relationship pairs whenever entities change, preserving existing decisions.
  // Entities that are rejected or merged are excluded. Entities differing only by case are
  // deduplicated (first occurrence wins), so "Fragile X syndrome" and "fragile X syndrome"
  // produce only one pair entry.
  useEffect(() => {
    if (!abstract) return
    const active = entities.filter(e => e.status !== 'rejected' && e.status !== 'merged')
    // Deduplicate by lowercase text, keeping first occurrence
    const seenLower = new Map()
    const deduped = []
    for (const e of active) {
      const key = e.text.toLowerCase()
      if (!seenLower.has(key)) { seenLower.set(key, true); deduped.push(e) }
    }
    const prelabelMap = {}
    for (const r of (abstract.prelabels.relationships || [])) {
      prelabelMap[`${r.subject.toLowerCase()}|||${r.object.toLowerCase()}`] = r
    }
    const existingMap = {}
    for (const r of relRef.current) {
      existingMap[`${r.subject.toLowerCase()}|||${r.object.toLowerCase()}`] = r
    }
    const pairs = []
    for (let i = 0; i < deduped.length; i++) {
      for (let j = i + 1; j < deduped.length; j++) {
        const e1 = deduped[i], e2 = deduped[j]
        const key = `${e1.text.toLowerCase()}|||${e2.text.toLowerCase()}`
        const keyRev = `${e2.text.toLowerCase()}|||${e1.text.toLowerCase()}`
        const existing = existingMap[key] || existingMap[keyRev]
        const prelabel = prelabelMap[key] || prelabelMap[keyRev]
        const same = inSameSentence(abstract.text, e1, e2)
        pairs.push(existing
          ? { ...existing, sameSentence: same }
          : {
              subject: e1.text, subjectStart: e1.start, subjectEnd: e1.end,
              object: e2.text, objectStart: e2.start, objectEnd: e2.end,
              type: prelabel?.type || '',
              direction: prelabel?.direction || 'positive',
              status: prelabel ? 'accepted' : 'rejected',
              sameSentence: same,
            }
        )
      }
    }
    pairs.sort((a, b) => (b.sameSentence ? 1 : 0) - (a.sameSentence ? 1 : 0))
    setRelationships(pairs)
  }, [entities, abstract])

  const save = useCallback(async () => {
    setSaving(true)
    try {
      await fetch(`/api/annotate/${id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          annotator_name: annotatorName,
          entities: entityRef.current,
          relationships: relRef.current,
          phase: 'completed',
        }),
      })
      setSavedMsg('Saved')
      setTimeout(() => setSavedMsg(''), 2000)
    } catch {
      setSavedMsg('Error')
    } finally {
      setSaving(false)
    }
  }, [id, annotatorName])

  const toggleEntityStatus = (idx) => {
    setEntities(prev => prev.map((e, i) =>
      i !== idx ? e : { ...e, status: e.status === 'rejected' ? 'accepted' : 'rejected', mergedInto: undefined }
    ))
    setPendingSelection(null)
    setEditingEntity(null)
  }

  const changeEntityType = (idx, type) => {
    setEntities(prev => prev.map((e, i) =>
      i !== idx ? e : { ...e, type, status: e.status === 'added' ? 'added' : 'edited' }
    ))
    setEditingEntity(null)
  }

  const deleteEntity = (idx) => {
    setEntities(prev => prev.filter((_, i) => i !== idx))
    setEditingEntity(null)
  }

  const mergeEntity = (idx, targetText) => {
    const sourceText = entities[idx].text
    setEntities(prev => prev.map((e, i) => {
      if (e.status === 'rejected' || e.status === 'merged') return e
      if (i === idx || e.text.toLowerCase() === sourceText.toLowerCase()) {
        return { ...e, status: 'merged', mergedInto: targetText }
      }
      return e
    }))
    setEditingEntity(null)
  }

  const addEntity = () => {
    if (!pendingSelection || !pendingSelection.selectedText.trim()) return
    const { start, end, selectedText } = pendingSelection
    setEntities(prev =>
      [...prev, { text: selectedText.trim(), type: newType, start, end, status: 'added' }]
        .sort((a, b) => a.start - b.start)
    )
    setPendingSelection(null)
  }

  const handleEntityClick = (entity) => {
    const idx = entities.findIndex(e => e.start === entity.start && e.end === entity.end)
    setEditingEntity({ entity, idx })
    setPendingSelection(null)
  }

  if (loading) return <div className="app"><div className="loading">Loading...</div></div>
  if (!abstract) return <div className="app"><div className="empty-state">Abstract not found.</div></div>

  const acceptedCount = entities.filter(e => e.status !== 'rejected' && e.status !== 'merged').length
  const acceptedRels = relationships.filter(r => r.status === 'accepted' || r.status === 'added').length

  return (
    <div className="app annotation-app">
      <header className="annotation-header">
        <button className="back-link" onClick={() => navigate('/annotate')}>&larr; Queue</button>
        <div className="annotation-title">
          Abstract #{id}{abstract.pmid ? ` · PMID ${abstract.pmid}` : ''}
        </div>
        <div className="annotation-actions">
          <span className="annotation-status">{saving ? 'Saving...' : savedMsg}</span>
          <button className="btn-primary btn-small" onClick={save}>Save</button>
        </div>
      </header>

      <div className="annotation-two-col">
        <div className="annotation-text-col">
          <AnnotatedText
            text={abstract.text}
            entities={entities}
            hoveredPair={hoveredPair}
            onEntityClick={handleEntityClick}
            onTextSelect={sel => { setPendingSelection(sel); setEditingEntity(null) }}
          />
          {pendingSelection && (
            <div className="selection-popover">
              <span className="sel-text">
                &ldquo;{pendingSelection.selectedText.slice(0, 50)}{pendingSelection.selectedText.length > 50 ? '...' : ''}&rdquo;
              </span>
              <select className="type-select" value={newType} onChange={e => setNewType(e.target.value)}>
                {ENTITY_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              <button className="btn-primary btn-small" onClick={addEntity}>Add</button>
              <button className="btn-ghost btn-small" onClick={() => setPendingSelection(null)}>Cancel</button>
            </div>
          )}
          {editingEntity && (
            <div className="entity-edit-popover">
              <span className="sel-text">{editingEntity.entity.text}</span>
              <select
                className="type-select"
                value={editingEntity.entity.type}
                onChange={e => changeEntityType(editingEntity.idx, e.target.value)}
              >
                {ENTITY_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              <select
                className="type-select"
                value=""
                onChange={e => { if (e.target.value) mergeEntity(editingEntity.idx, e.target.value) }}
                title="Mark this span as a duplicate of another entity"
              >
                <option value="">Same as...</option>
                {entities
                  .filter((e, i) => i !== editingEntity.idx && e.status !== 'rejected' && e.status !== 'merged')
                  .map(e => <option key={`${e.start}-${e.end}`} value={e.text}>{e.text}</option>)
                }
              </select>
              <button className="btn-danger btn-small" onClick={() => deleteEntity(editingEntity.idx)}>Delete</button>
              <button className="btn-ghost btn-small" onClick={() => setEditingEntity(null)}>Close</button>
            </div>
          )}
        </div>

        <div className="annotation-right-col">
          <div className="annotation-sidebar">
            <div className="sidebar-header">
              <span className="sidebar-title">Entities ({acceptedCount})</span>
            </div>
            <div className="sidebar-hint">Click an entity in the text to change its type or merge duplicates.</div>
            <div className="entity-list-panel">
              {entities.map((e, i) => {
                const variants = entities.filter(
                  (other, j) => j !== i && other.status === 'merged' && other.mergedInto === e.text
                )
                return (
                  <div key={i} className={`entity-panel-item ${e.status === 'rejected' ? 'entity-rejected' : ''} ${e.status === 'merged' ? 'entity-merged' : ''}`}>
                    <button
                      className={`status-toggle ${e.status === 'rejected' ? 'toggle-rejected' : e.status === 'merged' ? 'toggle-merged' : 'toggle-accepted'}`}
                      onClick={() => toggleEntityStatus(i)}
                      title={e.status === 'merged' ? `Unmerge from "${e.mergedInto}"` : e.status === 'rejected' ? 'Accept' : 'Reject'}
                    >
                      {e.status === 'rejected' ? '✗' : e.status === 'merged' ? '=' : '✓'}
                    </button>
                    <span className="entity-panel-text" onClick={() => handleEntityClick(e)}>
                      {e.text}
                    </span>
                    {e.status === 'merged'
                      ? <span className="merged-into">= {e.mergedInto}</span>
                      : <span className={`type-badge type-${e.type.toLowerCase()}`}>{e.type}</span>
                    }
                    {e.status === 'added' && <span className="added-marker">+</span>}
                    {variants.length > 0 && (
                      <span className="variant-badge" title={variants.map(v => v.text).join(', ')}>
                        +{variants.length}
                      </span>
                    )}
                  </div>
                )
              })}
              {entities.length === 0 && <div className="muted" style={{ padding: '0.5rem' }}>No entities</div>}
            </div>
          </div>

          <div className="rel-section">
            <div className="rel-section-header">
              <span className="sidebar-title">Relationships</span>
              <span className="rel-section-meta">
                {acceptedRels} of {relationships.length} with relationship
              </span>
            </div>
            <RelationshipList
              text={abstract.text}
              pairs={relationships}
              setPairs={setRelationships}
              onHoverPair={setHoveredPair}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Relationship List (all pairs on one page) ────────────────────────────────

function RelationshipList({ text, pairs, setPairs, onHoverPair }) {
  if (pairs.length === 0) {
    return <div className="muted" style={{ padding: '1rem 0' }}>No entity pairs to annotate.</div>
  }

  const updatePair = (idx, updates) => {
    setPairs(prev => prev.map((p, i) => i === idx ? { ...p, ...updates } : p))
  }

  return (
    <div className="rel-list">
      {pairs.map((pair) => {
        const idx = pairs.indexOf(pair)
        return (
          <RelationshipRow
            key={`${pair.subject}|||${pair.object}`}
            pair={pair}
            onUpdate={updates => updatePair(idx, updates)}
            onHoverPair={onHoverPair}
          />
        )
      })}
    </div>
  )
}

function RelationshipRow({ pair, onUpdate, onHoverPair }) {
  const swap = () => onUpdate({
    subject: pair.object, subjectStart: pair.objectStart, subjectEnd: pair.objectEnd,
    object: pair.subject, objectStart: pair.subjectStart, objectEnd: pair.subjectEnd,
  })

  return (
    <div
      className="rel-row"
      onMouseEnter={() => onHoverPair(pair)}
      onMouseLeave={() => onHoverPair(null)}
    >
      <div className="rel-entity-direction">
        <span className="rel-subject">{pair.subject}</span>
        <span className="rel-dir-arrow">→</span>
        <span className="rel-object">{pair.object}</span>
        <button className="swap-btn" onClick={swap} title="Swap direction">⇄</button>
      </div>
      <div className="rel-row-controls">
        <select
          className="type-select"
          value={pair.type || ''}
          onChange={e => onUpdate({ type: e.target.value, status: e.target.value ? 'accepted' : 'rejected' })}
        >
          <option value="">no relationship</option>
          {REL_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
        {pair.type && (
          <select
            className="type-select"
            value={pair.direction || 'positive'}
            onChange={e => onUpdate({ direction: e.target.value })}
            title="Polarity: whether the relationship holds as stated (positive) or is negated (negative)"
          >
            <option value="positive">positive</option>
            <option value="negative">negative</option>
            <option value="unknown">unknown</option>
          </select>
        )}
      </div>
    </div>
  )
}

// ─── AnnotatedText ────────────────────────────────────────────────────────────

function AnnotatedText({ text, entities, hoveredPair, onEntityClick, onTextSelect }) {
  const containerRef = useRef(null)

  const visible = entities
    .filter(e => e.status !== 'rejected')
    .sort((a, b) => a.start - b.start)

  // Build hover-only spans: text occurrences not covered by any entity span
  const hoverSpans = []
  if (hoveredPair) {
    const queries = [
      { q: hoveredPair.subject, cls: 'entity-hl-hov-subject' },
      { q: hoveredPair.object,  cls: 'entity-hl-hov-object' },
    ]
    for (const { q, cls } of queries) {
      for (const occ of findAllOccurrences(text, q)) {
        const covered = visible.some(e => e.start < occ.end && e.end > occ.start)
        if (!covered) hoverSpans.push({ t: 'hover', cls, start: occ.start, end: occ.end })
      }
    }
  }

  const allSpans = [
    ...visible.map(e => ({ t: 'entity', e, start: e.start, end: e.end })),
    ...hoverSpans,
  ].sort((a, b) => a.start - b.start)

  const segments = []
  let pos = 0
  for (const span of allSpans) {
    if (span.start < pos) continue
    if (span.start > pos) segments.push({ t: 'text', s: text.slice(pos, span.start) })
    segments.push({ ...span, s: text.slice(span.start, span.end) })
    pos = span.end
  }
  if (pos < text.length) segments.push({ t: 'text', s: text.slice(pos) })

  function getOffset(node, offset) {
    if (!containerRef.current) return 0
    let total = 0
    const walker = document.createTreeWalker(containerRef.current, NodeFilter.SHOW_TEXT)
    while (walker.nextNode()) {
      if (walker.currentNode === node) return total + offset
      total += walker.currentNode.length
    }
    return offset
  }

  function handleMouseUp() {
    const sel = window.getSelection()
    if (!sel || sel.isCollapsed || !containerRef.current) return
    const range = sel.getRangeAt(0)
    if (!containerRef.current.contains(range.commonAncestorContainer)) return
    const start = getOffset(range.startContainer, range.startOffset)
    const end = getOffset(range.endContainer, range.endOffset)
    sel.removeAllRanges()
    if (end > start) onTextSelect({ start, end, selectedText: text.slice(start, end) })
  }

  return (
    <div ref={containerRef} className="annotated-text" onMouseUp={handleMouseUp}>
      {segments.map((seg, i) => {
        if (seg.t === 'text') return <span key={i}>{seg.s}</span>
        if (seg.t === 'hover') return <mark key={i} className={`entity-hl ${seg.cls}`}>{seg.s}</mark>
        const { e } = seg
        const typeCls = `entity-hl entity-hl-${e.type.toLowerCase()}`
        const statusCls = e.status === 'added' ? ' entity-hl-added' : e.status === 'edited' ? ' entity-hl-edited' : ''
        const eText = e.text.toLowerCase()
        const eMergedInto = e.mergedInto ? e.mergedInto.toLowerCase() : null
        const hovSubj = hoveredPair && (eText === hoveredPair.subject.toLowerCase() || eMergedInto === hoveredPair.subject.toLowerCase())
        const hovObj = hoveredPair && (eText === hoveredPair.object.toLowerCase() || eMergedInto === hoveredPair.object.toLowerCase())
        const hovCls = hovSubj ? ' entity-hl-hov-subject' : hovObj ? ' entity-hl-hov-object' : ''
        return (
          <mark
            key={i}
            className={typeCls + statusCls + hovCls}
            onClick={ev => { ev.stopPropagation(); onEntityClick(e) }}
            title={`${e.type} — click to edit`}
          >
            {seg.s}
          </mark>
        )
      })}
    </div>
  )
}

// ─── ContextText ──────────────────────────────────────────────────────────────

function ContextText({ context, contextStart, entity1, entity2 }) {
  const marks = [
    { start: entity1.start - contextStart, end: entity1.end - contextStart, cls: 'ctx-e1' },
    { start: entity2.start - contextStart, end: entity2.end - contextStart, cls: 'ctx-e2' },
  ].filter(m => m.start >= 0 && m.end <= context.length && m.start < m.end)
   .sort((a, b) => a.start - b.start)

  const segments = []
  let pos = 0
  for (const m of marks) {
    if (m.start < pos) continue
    if (m.start > pos) segments.push({ t: 'text', s: context.slice(pos, m.start) })
    segments.push({ t: 'mark', s: context.slice(m.start, m.end), cls: m.cls })
    pos = m.end
  }
  if (pos < context.length) segments.push({ t: 'text', s: context.slice(pos) })

  return (
    <p className="context-text">
      {segments.map((s, i) =>
        s.t === 'text'
          ? <span key={i}>{s.s}</span>
          : <mark key={i} className={s.cls}>{s.s}</mark>
      )}
    </p>
  )
}

// ─── Shared formatters ────────────────────────────────────────────────────────

function formatType(type) {
  return type.replace(/_/g, ' ')
}

function confidenceTier(confidence) {
  if (confidence >= 0.9) return 'high'
  if (confidence >= 0.5) return 'medium'
  return 'low'
}

export default App
