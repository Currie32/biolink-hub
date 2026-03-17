import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import './App.css'

function parseRoute() {
  const path = window.location.pathname
  if (path.startsWith('/type/')) {
    return { view: 'type', type: decodeURIComponent(path.slice(6)) }
  }
  if (path.startsWith('/entity/')) {
    return { view: 'entity', id: decodeURIComponent(path.slice(8)) }
  }
  return { view: 'home' }
}

function navigate(path) {
  window.history.pushState(null, '', path)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [entity, setEntity] = useState(null)
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)
  const [typeFilter, setTypeFilter] = useState(null)
  const [typeEntities, setTypeEntities] = useState([])

  // Handle URL routing
  useEffect(() => {
    const handleRoute = () => {
      const route = parseRoute()
      if (route.view === 'type') {
        setEntity(null)
        setQuery('')
        setShowDropdown(false)
        setTypeFilter(route.type)
        setLoading(true)
        fetch(`/api/entities?type=${encodeURIComponent(route.type)}`)
          .then(r => r.json())
          .then(data => { setTypeEntities(data); setLoading(false) })
          .catch(() => setLoading(false))
      } else if (route.view === 'entity') {
        setShowDropdown(false)
        setTypeFilter(null)
        setTypeEntities([])
        setLoading(true)
        fetch(`/api/entity/${encodeURIComponent(route.id)}`)
          .then(r => r.json())
          .then(data => { setEntity(data); setLoading(false) })
          .catch(() => setLoading(false))
      } else {
        setEntity(null)
        setTypeFilter(null)
        setTypeEntities([])
      }
    }

    handleRoute()
    window.addEventListener('popstate', handleRoute)
    return () => window.removeEventListener('popstate', handleRoute)
  }, [])

  useEffect(() => {
    fetch('/api/stats').then(r => r.json()).then(setStats).catch(() => {})
  }, [])

  // Debounced search
  useEffect(() => {
    if (!query.trim()) {
      setResults([])
      setShowDropdown(false)
      return
    }
    const timer = setTimeout(() => {
      fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then(r => r.json())
        .then(data => {
          const seen = new Map()
          for (const r of data) {
            const key = `${r.name.toLowerCase()}::${r.type}`
            const existing = seen.get(key)
            if (!existing || (r.description && !existing.description)) {
              seen.set(key, r)
            }
          }
          setResults([...seen.values()])
          setShowDropdown(true)
        })
        .catch(() => {})
    }, 150)
    return () => clearTimeout(timer)
  }, [query])

  const selectEntity = useCallback((id) => {
    setShowDropdown(false)
    navigate(`/entity/${encodeURIComponent(id)}`)
  }, [])

  const selectType = useCallback((type) => {
    setQuery('')
    navigate(`/type/${encodeURIComponent(type)}`)
  }, [])

  const goHome = useCallback(() => {
    setQuery('')
    navigate('/')
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1 onClick={goHome} style={{ cursor: 'pointer' }}>
          BioLink Hub
        </h1>
        <p>Open Biomedical Knowledge Explorer</p>
      </header>

      <SearchBar
        query={query}
        setQuery={setQuery}
        results={results}
        showDropdown={showDropdown}
        setShowDropdown={setShowDropdown}
        onSelect={selectEntity}
      />

      {!entity && !loading && !typeFilter && stats && !query && (
        <StatsBar stats={stats} onTypeClick={selectType} />
      )}

      {loading && <div className="loading">Loading...</div>}

      {!entity && !loading && typeFilter && (
        <EntityList
          type={typeFilter}
          entities={typeEntities}
          onSelect={selectEntity}
          onBack={goHome}
        />
      )}

      {entity && !loading && (
        <EntityDetail entity={entity} onNavigate={selectEntity} />
      )}
    </div>
  )
}


function SearchBar({ query, setQuery, results, showDropdown, setShowDropdown, onSelect }) {
  const containerRef = useRef(null)
  const inputRef = useRef(null)
  const [highlightIdx, setHighlightIdx] = useState(-1)

  useEffect(() => {
    const handleClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [setShowDropdown])

  useEffect(() => { setHighlightIdx(-1) }, [results])

  const handleKeyDown = (e) => {
    if (!showDropdown || results.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlightIdx(i => Math.min(i + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlightIdx(i => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && highlightIdx >= 0) {
      e.preventDefault()
      onSelect(results[highlightIdx].id)
    } else if (e.key === 'Escape') {
      setShowDropdown(false)
    }
  }

  return (
    <div className="search-container" ref={containerRef}>
      <input
        ref={inputRef}
        className="search-input"
        type="text"
        placeholder="Search genes, diseases, drugs, pathways..."
        value={query}
        onChange={e => { setQuery(e.target.value); setShowDropdown(true) }}
        onFocus={() => { if (results.length > 0) setShowDropdown(true) }}
        onKeyDown={handleKeyDown}
        autoFocus
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
      {!query && (
        <p className="search-hint">
          Try: SNCA, APOE, Alzheimer, neuroinflammation, dopamine
        </p>
      )}
    </div>
  )
}


function StatsBar({ stats, onTypeClick }) {
  const sorted = Object.entries(stats.entities || {}).sort((a, b) => b[1] - a[1])
  return (
    <div className="stats-section">
      <div className="stats-grid">
        {sorted.map(([type, count]) => (
          <div
            key={type}
            className="stat-chip stat-chip-clickable"
            onClick={() => onTypeClick(type)}
          >
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
          <div
            key={e.id}
            className="entity-list-item"
            onClick={() => onSelect(e.id)}
          >
            <div className="entity-list-name">{e.name}</div>
            {e.description && (
              <div className="entity-list-desc">
                {e.description.length > 120 ? e.description.slice(0, 120) + '...' : e.description}
              </div>
            )}
            <div className="entity-list-id">{e.id}</div>
          </div>
        ))}
        {entities.length === 0 && (
          <div className="empty-state">No entities found.</div>
        )}
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
      const key = r.type
      if (!groups[key]) groups[key] = []
      groups[key].push(r)
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
            <div className="detail-section">
              <h3>Description</h3>
              <p>{entity.description}</p>
            </div>
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
                  <span key={db} className="ext-id-chip">
                    <strong>{db}:</strong> {id}
                  </span>
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
                      <a
                        className="rel-entity-link"
                        onClick={() => onNavigate(r.target_id === entity.id ? r.source_id : r.target_id)}
                      >
                        {r.related_name}
                      </a>
                      {r.confidence != null && (
                        <span className={`confidence-badge confidence-${confidenceTier(r.confidence)}`}>
                          {Math.round(r.confidence * 100)}%
                        </span>
                      )}
                      {r.evidence_count > 0 && (
                        <button
                          className="evidence-toggle"
                          onClick={(e) => { e.stopPropagation(); toggleEvidence(r.id) }}
                        >
                          {r.evidence_count} paper{r.evidence_count !== 1 ? 's' : ''}
                          {expandedRel === r.id ? ' \u25B4' : ' \u25BE'}
                        </button>
                      )}
                    </div>
                    {expandedRel === r.id && evidence[r.id] && (
                      <div className="evidence-list">
                        {evidence[r.id].map((ev, j) => (
                          <div key={j} className="evidence-item">
                            <a className="evidence-title" href={ev.source_url} target="_blank" rel="noopener noreferrer">
                              {ev.paper_title}
                            </a>
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


function formatType(type) {
  return type.replace(/_/g, ' ')
}

function confidenceTier(confidence) {
  if (confidence >= 0.9) return 'high'
  if (confidence >= 0.5) return 'medium'
  return 'low'
}

export default App
