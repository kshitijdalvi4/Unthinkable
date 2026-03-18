import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, Sparkles, Search, CheckCircle, Plus, X, ExternalLink,
  ChevronRight, Briefcase, AlertTriangle, Loader2, MessageCircle,
  Users, MapPin, Bot, LogOut, RefreshCw
} from 'lucide-react'
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google'
import { jwtDecode } from 'jwt-decode'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '668900299069-skpi9fp6k39ao7leh4ojv637gqkun4id.apps.googleusercontent.com'

// ---------- Types ----------
interface KnowledgeBase {
  name: string;
  email: string;
  skills: string[];
  experience_years: number;
  phone: string;
  expected_salary: string;
  current_ctc: string;
  notice_period: string;
}

interface SuggestedRole { role_name: string; reasoning: string }
interface CrawledJob { title: string; company: string; url: string; source: string; description: string }
interface FormQuestion { question_identifier: string; question_text: string; proposed_answer: string; is_unknown: boolean }

// ======================================================
// SHARED COMPONENTS
// ======================================================
// ======================================================
// LOGIN SCREEN
// ======================================================
function LoginScreen({ onLogin }: { onLogin: (data: any) => void }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSuccess(credentialResponse: any) {
    setLoading(true); setError('')
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 15000)
      const resp = await fetch(`${API}/api/auth/google/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: credentialResponse.credential }),
        signal: controller.signal
      })
      clearTimeout(timeout)
      if (!resp.ok) {
        const errText = await resp.text()
        throw new Error(`Server error ${resp.status}: ${errText}`)
      }
      const data = await resp.json()
      onLogin(data)
    } catch (e: any) {
      if (e.name === 'AbortError') {
        setError('Request timed out. Backend may be starting up — please try again in 30s.')
      } else {
        setError(`Authentication failed: ${e.message}`)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold tracking-tight mb-3">
          Land your next job,{' '}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-400">
            frictionlessly.
          </span>
        </h1>
        <p className="text-zinc-400 max-w-md mx-auto">
          Sign in with Google to manage your resume and automate your job applications.
        </p>
      </div>

      <div className="bg-zinc-900/60 border border-zinc-800 p-8 rounded-3xl backdrop-blur-xl flex flex-col items-center gap-6">
        <div className="w-16 h-16 bg-emerald-500/10 rounded-full flex items-center justify-center">
          <Bot className="text-emerald-400" size={32} />
        </div>
        
        {loading ? (
          <div className="flex flex-col items-center gap-2">
            <Loader2 className="animate-spin text-emerald-400" />
            <p className="text-xs text-zinc-500">Syncing with Google...</p>
          </div>
        ) : (
          <GoogleLogin onSuccess={handleSuccess} onError={() => setError('Google Login Failed')} />
        )}

        {error && <p className="text-red-400 text-xs">{error}</p>}
      </div>
    </div>
  )
}

function Stepper({ step }: { step: number }) {
  const STEPS = ['Upload', 'Onboarding', 'Roles & Search', 'Jobs']
  return (
    <div className="flex items-center justify-center gap-2 mb-10">
      {STEPS.map((label, i) => (
        <div key={label} className="flex items-center gap-2">
          <div className={`flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold transition-all
            ${i < step ? 'bg-emerald-500 text-black' : i === step ? 'bg-emerald-500/20 border border-emerald-500 text-emerald-400' : 'bg-zinc-800 text-zinc-500'}`}>
            {i < step ? <CheckCircle size={14} /> : i + 1}
          </div>
          <span className={`text-xs hidden sm:block ${i === step ? 'text-emerald-400 font-semibold' : 'text-zinc-500'}`}>{label}</span>
          {i < STEPS.length - 1 && <div className={`w-8 h-px ${i < step ? 'bg-emerald-500' : 'bg-zinc-700'}`} />}
        </div>
      ))}
    </div>
  )
}

// ======================================================
// JOB HUNT TAB — Step 1: Upload (or Update)
// ======================================================
function UploadStep({ candidateId, onUploaded }: { candidateId: string, onUploaded: (id: string, name: string, data: any) => void }) {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleFile(file: File) {
    if (!file.name.endsWith('.pdf')) { setError('Only PDF files are supported.'); return }
    setLoading(true); setError('')
    const formData = new FormData(); 
    formData.append('file', file)
    if (candidateId) formData.append('candidate_id', candidateId)

    try {
      const resp = await fetch(`${API}/upload-resume/`, { method: 'POST', body: formData })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      onUploaded(data.candidate_id, data.data?.name || 'Candidate', data.data)
    } catch (e: any) { setError(e.message || 'Upload failed') }
    finally { setLoading(false) }
  }
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files[0]; if (file) handleFile(file)
  }, [])

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-lg mx-auto">
      <div onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}
        className={`border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all duration-300
          ${dragging ? 'border-emerald-400 bg-emerald-500/10' : 'border-zinc-700 hover:border-emerald-500/60 bg-zinc-900/60'}`}
        onClick={() => document.getElementById('resume-input')?.click()}>
        <input id="resume-input" type="file" accept=".pdf" className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />
        {loading ? (
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="text-emerald-400 animate-spin" size={40} />
            <p className="text-zinc-400">Uploading & analyzing resume…</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 bg-emerald-500/10 rounded-full flex items-center justify-center">
              <Upload className="text-emerald-400" size={28} />
            </div>
            <div><p className="text-white font-semibold text-lg">Drop your resume here</p>
              <p className="text-zinc-500 text-sm mt-1">or click to browse — PDF only</p></div>
          </div>
        )}
      </div>
      {error && <p className="mt-3 text-red-400 text-sm text-center">{error}</p>}
    </motion.div>
  )
}

function OnboardingStep({ candidateId, profile, onComplete }: { candidateId: string, profile: KnowledgeBase, onComplete: (updated: KnowledgeBase) => void }) {
  const [localProfile, setLocalProfile] = useState<KnowledgeBase>(profile)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSave() {
    setLoading(true); setError('')
    try {
      const resp = await fetch(`${API}/api/v3/update-profile`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_id: candidateId, profile: localProfile })
      })
      if (!resp.ok) throw new Error(await resp.text())
      onComplete(localProfile)
    } catch (e: any) { setError(e.message) }
    finally { setLoading(false) }
  }

  return (
    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="max-w-2xl mx-auto bg-zinc-900/60 border border-zinc-800 rounded-3xl p-8 backdrop-blur-xl">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-emerald-500/20 text-emerald-400 rounded-xl flex items-center justify-center">
          <Bot size={22} />
        </div>
        <div>
          <h2 className="text-xl font-bold">Smart Onboarding</h2>
          <p className="text-zinc-500 text-xs">Confirm your profile details for intelligent form filling</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-8">
        <div>
          <label className="text-xs text-zinc-500 mb-1.5 block">Full Name</label>
          <input value={localProfile.name} onChange={e => setLocalProfile({...localProfile, name: e.target.value})} 
            className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white text-sm focus:border-emerald-500 focus:outline-none" />
        </div>
        <div>
          <label className="text-xs text-zinc-500 mb-1.5 block">Email Address</label>
          <input value={localProfile.email} onChange={e => setLocalProfile({...localProfile, email: e.target.value})} 
            className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white text-sm focus:border-emerald-500 focus:outline-none" />
        </div>
        <div>
          <label className="text-xs text-zinc-500 mb-1.5 block">Phone Number</label>
          <input value={localProfile.phone} onChange={e => setLocalProfile({...localProfile, phone: e.target.value})} 
            className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white text-sm focus:border-emerald-500 focus:outline-none" />
        </div>
        <div>
          <label className="text-xs text-zinc-500 mb-1.5 block">Total Experience (Years)</label>
          <input type="number" value={localProfile.experience_years} onChange={e => setLocalProfile({...localProfile, experience_years: Number(e.target.value)})} 
            className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white text-sm focus:border-emerald-500 focus:outline-none" />
        </div>
        <div>
          <label className="text-xs text-zinc-500 mb-1.5 block">Current CTC</label>
          <input value={localProfile.current_ctc} onChange={e => setLocalProfile({...localProfile, current_ctc: e.target.value})} placeholder="e.g. 15,00,000"
            className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white text-sm focus:border-emerald-500 focus:outline-none" />
        </div>
        <div>
          <label className="text-xs text-zinc-500 mb-1.5 block">Expected CTC</label>
          <input value={localProfile.expected_salary} onChange={e => setLocalProfile({...localProfile, expected_salary: e.target.value})} placeholder="e.g. 25,00,000"
            className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white text-sm focus:border-emerald-500 focus:outline-none" />
        </div>
        <div className="md:col-span-2">
          <label className="text-xs text-zinc-500 mb-1.5 block">Skills (comma separated)</label>
          <input value={localProfile.skills.join(', ')} onChange={e => setLocalProfile({...localProfile, skills: e.target.value.split(',').map(s => s.trim()).filter(Boolean)})} 
            className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white text-sm focus:border-emerald-500 focus:outline-none" />
        </div>
      </div>

      <button onClick={handleSave} disabled={loading}
        className="w-full bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-black font-bold py-3.5 rounded-2xl flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/20 transition-all">
        {loading ? <Loader2 size={20} className="animate-spin" /> : <>Save & Continue <ChevronRight size={18} /></>}
      </button>
      {error && <p className="text-red-400 text-xs text-center mt-4">{error}</p>}
    </motion.div>
  )
}

// ======================================================
// JOB HUNT TAB — Step 2: Roles + Location + Work Type
// ======================================================
const COUNTRIES = ['Worldwide', 'India', 'United States', 'United Kingdom', 'Canada', 'Australia', 'Germany', 'Singapore']
const WORK_TYPES = ['Any', 'Remote', 'Hybrid', 'In-Office']

function RoleSelectionStep({ candidateId, candidateName, onConfirm }:
  { candidateId: string; candidateName: string; onConfirm: (roles: string[], location: string, cities: string[], workType: string) => void }) {
  const [suggestedRoles, setSuggestedRoles] = useState<SuggestedRole[]>([])
  const [selectedRoles, setSelectedRoles] = useState<string[]>([])
  const [customInput, setCustomInput] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  // Location state
  const [location, setLocation] = useState('Worldwide')
  const [cityInput, setCityInput] = useState('')
  const [cities, setCities] = useState<string[]>([])
  const [workType, setWorkType] = useState('Any')

  useEffect(() => {
    fetch(`${API}/api/v2/suggest-roles/`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate_id: candidateId })
    }).then(r => r.json()).then(data => {
      setSuggestedRoles(data.suggested_roles || [])
      setSelectedRoles((data.suggested_roles || []).map((r: SuggestedRole) => r.role_name))
    }).catch(e => setError(e.message)).finally(() => setLoading(false))
  }, [candidateId])

  function addCity() { const c = cityInput.trim(); if (c && !cities.includes(c)) setCities(p => [...p, c]); setCityInput('') }
  function addCustomRole() { const r = customInput.trim(); if (r && !selectedRoles.includes(r)) setSelectedRoles(p => [...p, r]); setCustomInput('') }
  const customRoles = selectedRoles.filter(r => !suggestedRoles.find(s => s.role_name === r))

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-2xl mx-auto space-y-4">
      <div className="text-center mb-6">
        <div className="inline-flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/30 rounded-full px-4 py-1.5 text-emerald-400 text-sm mb-3">
          <CheckCircle size={14} /> {candidateName} — resume loaded
        </div>
        <h2 className="text-2xl font-bold text-white">Configure Your Job Search</h2>
      </div>

      {loading ? (
        <div className="flex justify-center py-12"><Loader2 className="text-emerald-400 animate-spin" size={32} /></div>
      ) : error ? (
        <p className="text-red-400 text-center">{error}</p>
      ) : (
        <>
          {/* AI Roles (2a) */}
          <div className="bg-zinc-900/70 border border-zinc-800 rounded-2xl p-5">
            <div className="flex items-center gap-2 text-xs text-zinc-400 mb-3"><Sparkles size={12} className="text-emerald-400" /> AI-Suggested Roles — click to toggle</div>
            <div className="flex flex-wrap gap-2">
              {suggestedRoles.map(r => {
                const active = selectedRoles.includes(r.role_name)
                return (
                  <button key={r.role_name} title={r.reasoning}
                    onClick={() => active ? setSelectedRoles(p => p.filter(x => x !== r.role_name)) : setSelectedRoles(p => [...p, r.role_name])}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all
                      ${active ? 'bg-emerald-500 text-black' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}`}>
                    {r.role_name}
                    {active && <X size={11} onClick={(e) => { e.stopPropagation(); setSelectedRoles(p => p.filter(x => x !== r.role_name)) }} />}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Custom Roles (2b) */}
          <div className="bg-zinc-900/70 border border-zinc-800 rounded-2xl p-5">
            <div className="flex items-center gap-2 text-xs text-zinc-400 mb-3"><Plus size={12} className="text-emerald-400" /> Add Your Own Roles</div>
            <div className="flex gap-2">
              <input value={customInput} onChange={e => setCustomInput(e.target.value)} 
                onKeyDown={e => { if(e.key === 'Enter') { e.preventDefault(); addCustomRole(); } }}
                placeholder="e.g. Product Manager, AI Researcher…"
                className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2 text-white placeholder-zinc-500 focus:outline-none focus:border-emerald-500 text-sm" />
              <button type="button" onClick={addCustomRole} className="bg-emerald-500 hover:bg-emerald-400 text-black font-semibold px-4 py-2 rounded-xl text-sm">Add</button>
            </div>
            {customRoles.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                {customRoles.map(r => (
                  <span key={r} className="flex items-center gap-1 bg-blue-500/20 text-blue-300 border border-blue-500/30 px-3 py-1 rounded-full text-sm">
                    {r} <X size={11} className="cursor-pointer" onClick={() => setSelectedRoles(p => p.filter(x => x !== r))} />
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Location & Work Type */}
          <div className="bg-zinc-900/70 border border-zinc-800 rounded-2xl p-5">
            <div className="flex items-center gap-2 text-xs text-zinc-400 mb-4"><MapPin size={12} className="text-emerald-400" /> Location & Work Preferences</div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="text-xs text-zinc-500 mb-1.5 block">Country</label>
                <select value={location} onChange={e => setLocation(e.target.value)}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-3 py-2.5 text-white text-sm focus:outline-none focus:border-emerald-500">
                  {COUNTRIES.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-zinc-500 mb-1.5 block">Work Type</label>
                <div className="flex gap-1.5">
                  {WORK_TYPES.map(t => (
                    <button key={t} onClick={() => setWorkType(t)}
                      className={`flex-1 py-2 rounded-xl text-xs font-medium transition-all border
                        ${workType === t ? 'bg-emerald-500 border-emerald-500 text-black' : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:border-zinc-500'}`}>
                      {t === 'Remote' ? '🌍' : t === 'Hybrid' ? '🏠' : t === 'In-Office' ? '🏢' : '⚡'} {t}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div>
              <label className="text-xs text-zinc-500 mb-1.5 block">Cities (optional — press Enter to add multiple)</label>
              <div className="flex gap-2 mb-2">
                <input value={cityInput} onChange={e => setCityInput(e.target.value)} 
                  onKeyDown={e => { if(e.key === 'Enter') { e.preventDefault(); addCity(); } }}
                  placeholder="e.g. Mumbai, Bangalore, London…"
                  className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2 text-white placeholder-zinc-500 focus:outline-none focus:border-emerald-500 text-sm" />
                <button type="button" onClick={addCity} className="bg-zinc-700 hover:bg-zinc-600 text-white px-4 py-2 rounded-xl text-sm">Add</button>
              </div>
              {cities.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {cities.map(c => (
                    <span key={c} className="flex items-center gap-1 bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-2.5 py-0.5 rounded-full text-xs">
                      📍 {c} <X size={10} className="cursor-pointer" onClick={() => setCities(p => p.filter(x => x !== c))} />
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between pt-2">
            <p className="text-zinc-500 text-sm">{selectedRoles.length} role{selectedRoles.length !== 1 ? 's' : ''} selected</p>
            <button type="button" disabled={selectedRoles.length === 0} onClick={() => onConfirm(selectedRoles, location, cities, workType)}
              className="flex items-center gap-2 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 text-black font-semibold px-6 py-3 rounded-xl transition-colors">
              Find Jobs <ChevronRight size={16} />
            </button>
          </div>
        </>
      )}
    </motion.div>
  )
}

// ======================================================
// JOB HUNT TAB — Step 3: Job Feed
// ======================================================
function JobCard({ job, onApply }: { job: CrawledJob; onApply: (job: CrawledJob) => void }) {
  const [hovered, setHovered] = useState(false)

  return (
    <motion.div layout initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}
      className="relative bg-zinc-900/80 border border-zinc-800 rounded-2xl p-6 hover:border-emerald-500/40 transition-all shadow-lg hover:shadow-emerald-500/5 group"
      onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
      
      <div className="absolute top-0 right-0 p-3 opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="w-8 h-8 bg-emerald-500 rounded-full flex items-center justify-center text-black shadow-lg">
          <Sparkles size={14} />
        </div>
      </div>

      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <h3 className="font-bold text-white text-lg leading-tight group-hover:text-emerald-400 transition-colors">{job.title}</h3>
          <div className="flex items-center gap-2 mt-1">
             <p className="text-zinc-400 text-sm font-medium">{job.company !== 'Unknown' ? job.company : 'Various Companies'}</p>
             <span className="w-1 h-1 bg-zinc-700 rounded-full" />
             <span className="text-zinc-500 text-xs">{job.source}</span>
          </div>
        </div>
      </div>

      <p className="text-zinc-500 text-xs mb-6 line-clamp-3 leading-relaxed">{job.description}</p>
      
      <div className="space-y-3 mb-6">
        <div className="flex items-center gap-2 text-[10px] text-zinc-500 uppercase tracking-widest font-bold">
          <CheckCircle size={10} className="text-emerald-500" /> Matches your profile
        </div>
        <div className="flex flex-wrap gap-1.5">
          {['Python', 'AI/ML', 'Remote'].map(tag => (
            <span key={tag} className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded text-[10px] font-bold">
              {tag}
            </span>
          ))}
        </div>
      </div>

      <div className="flex gap-2">
        <button type="button" onClick={() => onApply(job)}
          className="flex-1 bg-emerald-500 hover:bg-emerald-400 text-black font-bold py-2.5 rounded-xl text-sm transition-all shadow-lg shadow-emerald-500/10 active:scale-95">
          One-Click Apply
        </button>
        <a href={job.url} target="_blank" rel="noreferrer"
          className="flex items-center justify-center w-11 h-10 rounded-xl border border-zinc-700 text-zinc-400 hover:text-white hover:border-zinc-500 transition-all">
          <ExternalLink size={16} />
        </a>
      </div>

      <AnimatePresence>
        {hovered && (
          <motion.div initial={{ opacity: 0, y: 10, filter: 'blur(5px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0)' }} exit={{ opacity: 0, scale: 0.95 }}
            className="absolute bottom-full left-0 right-0 mb-3 bg-zinc-950/95 border border-emerald-500/20 rounded-2xl p-5 z-50 shadow-2xl backdrop-blur-md">
            <p className="text-xs font-bold text-emerald-400 mb-3 flex items-center gap-1.5">
              <Bot size={13} /> Intelligent Form Filling
            </p>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center text-zinc-400">
                <span>Personal Info</span>
                <span className="flex items-center gap-1 text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                  <CheckCircle size={10} /> Ready
                </span>
              </div>
              <div className="flex justify-between items-center text-zinc-400">
                <span>Experience</span>
                <span className="flex items-center gap-1 text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                  <CheckCircle size={10} /> Ready
                </span>
              </div>
              <div className="flex justify-between items-center text-zinc-400">
                <span>Salary Details</span>
                <span className="flex items-center gap-1 text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                  <CheckCircle size={10} /> Ready
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

function JobFeedStep({ roles, location, cities, workType, onApply, jobs, loading, error, onSearch }:
  { 
    roles: string[]; location: string; cities: string[]; workType: string; 
    onApply: (job: CrawledJob) => void;
    jobs: CrawledJob[]; loading: boolean; error: string; onSearch: () => void 
  }) {

  useEffect(() => {
    if (jobs.length === 0 && !loading && !error) {
      onSearch()
    }
  }, [jobs.length, loading, error, onSearch])

  const locationLabel = cities.length ? cities.join(', ') : location

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-white">Live Opportunities</h2>
        <p className="text-zinc-400 mt-1 text-sm">
          {loading ? `Crawling ${roles.join(', ')} jobs in ${locationLabel}…`
            : `Found ${jobs.length} openings · ${locationLabel} · ${workType}`}
        </p>
      </div>
      {loading ? (
        <div className="flex flex-col items-center gap-4 py-16">
          <div className="relative w-16 h-16">
            <div className="absolute inset-0 rounded-full border-2 border-emerald-500/20 animate-ping" />
            <div className="absolute inset-2 rounded-full bg-emerald-500/10 flex items-center justify-center">
              <Search className="text-emerald-400" size={20} />
            </div>
          </div>
          <p className="text-zinc-400">Crawling LinkedIn, Indeed, Jooble…</p>
        </div>
      ) : error ? (
        <p className="text-red-400 text-center">{error}</p>
      ) : jobs.length === 0 ? (
        <div className="text-center py-12 text-zinc-500">No jobs found. Try broader roles or different locations.</div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {jobs.map((job, i) => <JobCard key={i} job={job} onApply={onApply} />)}
        </div>
      )}
    </motion.div>
  )
}

// ======================================================
// JOB HUNT TAB — Step 4: Apply Modal
// ======================================================
function ApplyModal({ candidateId, job, onClose, onIdSync }:
  { candidateId: string; job: CrawledJob; onClose: () => void; onIdSync: (id: string) => void }) {
  const [loading, setLoading] = useState(true)
  const [questions, setQuestions] = useState<FormQuestion[]>([])
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [threadId, setThreadId] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const [autoSubmitting, setAutoSubmitting] = useState(false)
  const [autoSubmitStatus, setAutoSubmitStatus] = useState('')
  const [credentials, setCredentials] = useState({ email: '', password: '' })

  const unknowns = questions.filter(q => q.is_unknown)

  useEffect(() => {
    fetch(`${API}/api/v2/apply-job/`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate_id: candidateId, selected_job_url: job.url, job_title: job.title })
    }).then(r => r.json()).then(data => {
      setThreadId(data.thread_id || '')
      if (data.candidate_id && data.candidate_id !== candidateId) {
        onIdSync(data.candidate_id)
      }
      if (data.status === 'closed') {
        setError('This job is no longer accepting responses on LinkedIn.');
        setTimeout(() => onClose(), 3000);
        return;
      }
      const qs: FormQuestion[] = data.form_questions || []
      setQuestions(qs)
      const init: Record<string, string> = {}
      qs.forEach(q => { init[q.question_identifier] = q.proposed_answer || '' })
      setAnswers(init)
    }).catch(e => setError(e.message)).finally(() => setLoading(false))
  }, [])

  async function handleApprove() {
    setSubmitting(true)
    try {
      await fetch(`${API}/api/v2/approve-application/`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadId, candidate_id: candidateId, answers })
      })
      setSubmitted(true)
    } catch (e: any) { setError(e.message) } finally { setSubmitting(false) }
  }

  async function handleAutoSubmit() {
    setAutoSubmitting(true)
    setAutoSubmitStatus('Launching browser agent...')
    try {
      const res = await fetch(`${API}/api/v3/auto-submit/`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidate_id: candidateId,
          job_url: job.url,
          job_title: job.title,
          credentials: credentials.email ? credentials : null
        })
      })
      const data = await res.json()
      if (res.ok) {
        setAutoSubmitStatus('Agent started! Monitoring progress...')
        const poll = setInterval(async () => {
          try {
            const sRes = await fetch(`${API}/api/v3/automation-status/${data.candidate_id || candidateId}`)
            const sData = await sRes.json()
            if (['completed', 'failed', 'action_required', 'closed'].includes(sData.status)) {
              if (sData.status === 'closed') {
                setAutoSubmitStatus('This job is no longer accepting responses.');
                setTimeout(() => {
                  onClose();
                }, 3000);
              } else {
                setAutoSubmitStatus(`${sData.status === 'completed' ? '✅' : '❌'} ${sData.message}`)
              }
              setAutoSubmitting(false)
              clearInterval(poll)
            } else {
              setAutoSubmitStatus(`⏳ ${sData.message}`)
            }
          } catch (e) { clearInterval(poll) }
        }, 3000)
      } else {
        setAutoSubmitStatus(`❌ Launch failed: ${data.detail}`)
        setAutoSubmitting(false)
      }
    } catch (e: any) {
      setAutoSubmitStatus(`❌ Fetch failed.`)
      setAutoSubmitting(false)
    }
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <motion.div initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }}
        className="bg-zinc-900 border border-zinc-800 w-full max-w-2xl max-h-[90vh] overflow-hidden rounded-3xl shadow-2xl flex flex-col"
        onClick={e => e.stopPropagation()}>
        
        <div className="px-6 py-4 border-b border-zinc-800 flex items-center justify-between bg-zinc-900/50 backdrop-blur-md sticky top-0 z-10">
          <div>
            <h3 className="text-white font-bold leading-tight">{job.company}</h3>
            <p className="text-zinc-400 text-xs">{job.title} · {job.source}</p>
          </div>
          <button type="button" onClick={onClose} className="text-zinc-500 hover:text-white transition-colors bg-zinc-800 p-1.5 rounded-lg">
            <X size={18} />
          </button>
        </div>

        <div className="p-6 overflow-y-auto custom-scrollbar">
          {submitted ? (
            <div className="bg-zinc-950 border border-zinc-800 rounded-2xl p-8 text-center">
              <div className="w-16 h-16 bg-emerald-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
                <CheckCircle className="text-emerald-500" size={32} />
              </div>
              <h4 className="text-xl font-bold mb-2">Application Logged!</h4>
              <p className="text-zinc-500 text-sm mb-8">Your info is stored. The agent is now ready to assist in the browser.</p>
              
              <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 text-left mb-8 max-w-sm mx-auto shadow-xl shadow-black/20">
                <h5 className="flex items-center gap-2 text-white text-sm font-bold mb-3 uppercase tracking-wider">
                   <Bot size={16} className="text-emerald-400" /> Auto-Submit
                </h5>
                <p className="text-zinc-400 text-[11px] mb-5 leading-relaxed">Launch agent to fill the form automatically.</p>
                <div className="space-y-3">
                  <input value={credentials.email} onChange={e => setCredentials({...credentials, email: e.target.value})}
                    placeholder="Platform Email" className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2 text-xs focus:border-emerald-500 focus:outline-none transition-colors" />
                  <input value={credentials.password} onChange={e => setCredentials({...credentials, password: e.target.value})}
                    type="password" placeholder="Platform Password" className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2 text-xs focus:border-emerald-500 focus:outline-none transition-colors" />
                  <button onClick={handleAutoSubmit} disabled={autoSubmitting}
                    className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-bold py-3 rounded-xl text-xs flex items-center justify-center gap-2 transition-all">
                    {autoSubmitting ? <Loader2 size={16} className="animate-spin" /> : 'Launch Agent'}
                  </button>
                </div>
                {autoSubmitStatus && <div className="mt-4 p-3 bg-zinc-950 rounded-xl border border-zinc-800 font-mono text-[10px] text-zinc-400 break-words">{autoSubmitStatus}</div>}
              </div>
              <div className="flex flex-col gap-3">
                <a href={job.url} target="_blank" rel="noreferrer" className="text-emerald-400 hover:text-emerald-300 font-medium text-xs underline">Open Link Manually</a>
                <button onClick={onClose} className="text-zinc-500 hover:text-white text-xs mt-4">Close Window</button>
              </div>
            </div>
          ) : loading ? (
            <div className="flex flex-col items-center gap-4 py-12">
              <Loader2 className="text-emerald-400 animate-spin" size={32} />
              <p className="text-zinc-400 text-sm">Preparing application profile…</p>
            </div>
          ) : error ? (
            <p className="text-red-400 text-center text-sm">{error}</p>
          ) : (
            <>
              {unknowns.length > 0 && (
                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-3 flex items-center gap-2 mb-5">
                  <AlertTriangle className="text-yellow-400 shrink-0" size={14} />
                  <p className="text-yellow-300 text-sm"><strong>{unknowns.length} fields</strong> need your input</p>
                </div>
              )}
              <div className="space-y-4">
                {questions.map(q => (
                  <div key={q.question_identifier} className={`rounded-xl p-4 border ${q.is_unknown ? 'bg-yellow-500/5 border-yellow-500/30' : 'bg-zinc-900 border-zinc-800'}`}>
                    <div className="flex items-start justify-between mb-2 gap-2">
                      <label className="text-sm font-medium text-zinc-300">{q.question_text}</label>
                      {q.is_unknown ? <span className="text-[10px] bg-yellow-500/20 text-yellow-300 px-2 py-0.5 rounded-full font-bold">MISSING</span> : <span className="text-[10px] bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-full font-bold">PRE-FILLED</span>}
                    </div>
                    <textarea value={answers[q.question_identifier] || ''} onChange={e => setAnswers(p => ({ ...p, [q.question_identifier]: e.target.value }))} rows={2} className="w-full bg-zinc-800/80 border border-zinc-700 focus:border-emerald-500 rounded-lg px-3 py-2 text-white text-sm resize-none focus:outline-none transition-colors" />
                  </div>
                ))}
              </div>
              <div className="flex gap-3 mt-6 pt-6 border-t border-zinc-800">
                <button type="button" onClick={onClose} className="flex-1 py-3 rounded-xl border border-zinc-700 text-zinc-400 hover:text-white text-sm">Cancel</button>
                <button type="button" onClick={handleApprove} disabled={submitting} className="flex-1 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-black font-semibold py-3 rounded-xl text-sm flex items-center justify-center gap-2">
                  {submitting ? <><Loader2 size={14} className="animate-spin" /> Submitting…</> : '✓ Approve & Log'}
                </button>
              </div>
            </>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}

// ======================================================
// HR MODE TAB — Job Description Matching (V1 feature)
// ======================================================
function HRModeTab({ candidateId }: { candidateId: string }) {
  console.log('HR Mode for:', candidateId)
  const [jd, setJd] = useState({ title: '', description: '', required_skills: '', experience_years: '0' })
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleMatch() {
    setLoading(true); setResults([]); setError('')
    try {
      const resp = await fetch(`${API}/match-job/`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: jd.title, description: jd.description,
          required_skills: jd.required_skills.split(',').map(s => s.trim()).filter(Boolean),
          experience_years: parseInt(jd.experience_years) || 0
        })
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      setResults(data.matches || [])
    } catch (e: any) { setError(e.message) } finally { setLoading(false) }
  }

  return (
    <div className="max-w-3xl mx-auto space-y-4">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white">HR Mode</h2>
        <p className="text-zinc-400 text-sm mt-1">Paste a Job Description and match it against uploaded resumes</p>
      </div>
      <div className="bg-zinc-900/70 border border-zinc-800 rounded-2xl p-5 space-y-3">
        <input placeholder="Job Title" value={jd.title} onChange={e => setJd(p => ({ ...p, title: e.target.value }))}
          className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white placeholder-zinc-500 text-sm focus:outline-none focus:border-emerald-500" />
        <textarea placeholder="Job Description…" value={jd.description} onChange={e => setJd(p => ({ ...p, description: e.target.value }))}
          rows={5} className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white placeholder-zinc-500 text-sm focus:outline-none focus:border-emerald-500 resize-none" />
        <input placeholder="Required Skills (comma separated)" value={jd.required_skills} onChange={e => setJd(p => ({ ...p, required_skills: e.target.value }))}
          className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white placeholder-zinc-500 text-sm focus:outline-none focus:border-emerald-500" />
        <div className="flex items-center gap-3">
          <input type="number" placeholder="Min. Years Exp." value={jd.experience_years} onChange={e => setJd(p => ({ ...p, experience_years: e.target.value }))}
            className="w-40 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white text-sm focus:outline-none focus:border-emerald-500" />
          <button type="button" onClick={handleMatch} disabled={!jd.title || loading}
            className="flex-1 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 text-black font-semibold py-2.5 rounded-xl text-sm flex items-center justify-center gap-2">
            {loading ? <><Loader2 size={14} className="animate-spin" />Matching…</> : <><Search size={14} />Match Resumes</>}
          </button>
        </div>
      </div>
      {error && <p className="text-red-400 text-sm text-center">{error}</p>}
      {results.length > 0 && (
        <div className="space-y-3">
          {results.map((r, i) => (
            <div key={i} className="bg-zinc-900/70 border border-zinc-800 rounded-2xl p-5">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-white">{r.candidate_name}</span>
                <span className={`text-sm font-bold ${r.match_score >= 70 ? 'text-emerald-400' : r.match_score >= 40 ? 'text-yellow-400' : 'text-red-400'}`}>
                  {r.match_score?.toFixed(0)}% match
                </span>
              </div>
              <p className="text-zinc-400 text-xs mb-2">{r.justification}</p>
              <div className="flex gap-2 flex-wrap">
                {(r.matched_skills || []).map((s: string) => (
                  <span key={s} className="text-xs bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-full">{s}</span>
                ))}
                {(r.missing_skills || []).map((s: string) => (
                  <span key={s} className="text-xs bg-zinc-700 text-zinc-400 px-2 py-0.5 rounded-full line-through">{s}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ======================================================
// RESUME CHAT TAB — RAG Q&A (V1 feature)
// ======================================================
function ResumeChatTab({ candidateId }: { candidateId: string }) {
  const [messages, setMessages] = useState<{ role: 'user' | 'ai'; text: string }[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  async function sendMessage() {
    const q = input.trim(); if (!q) return
    setMessages(p => [...p, { role: 'user', text: q }])
    setInput(''); setLoading(true)
    try {
      const resp = await fetch(`${API}/chat-resume/`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_id: candidateId, question: q })
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json()
      setMessages(p => [...p, { role: 'ai', text: data.answer || 'No answer received.' }])
    } catch (e: any) {
      setMessages(p => [...p, { role: 'ai', text: `Error: ${e.message}` }])
    } finally { setLoading(false) }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white">Resume Chat</h2>
        <p className="text-zinc-400 text-sm mt-1">Ask anything about the uploaded resume</p>
      </div>
      {!candidateId ? (
        <div className="text-center py-16 text-zinc-500">
          <MessageCircle size={40} className="mx-auto mb-3 opacity-30" />
          <p>Upload a resume in the Job Hunt tab first</p>
        </div>
      ) : (
        <>
          <div className="bg-zinc-900/70 border border-zinc-800 rounded-2xl p-4 min-h-64 max-h-96 overflow-y-auto space-y-3 mb-4">
            {messages.length === 0 && (
              <p className="text-zinc-600 text-sm text-center py-8">Try: "What are the candidate's top skills?" or "How many years of experience do they have?"</p>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm
                  ${m.role === 'user' ? 'bg-emerald-500 text-black' : 'bg-zinc-800 text-zinc-200'}`}>
                  {m.text}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-zinc-800 px-4 py-2.5 rounded-2xl flex items-center gap-2">
                  <Loader2 size={12} className="animate-spin text-zinc-400" />
                  <span className="text-zinc-400 text-sm">Thinking…</span>
                </div>
              </div>
            )}
          </div>
          <div className="flex gap-2">
            <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && !loading && sendMessage()}
              placeholder="Ask about the resume…"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-white placeholder-zinc-500 text-sm focus:outline-none focus:border-emerald-500" />
            <button onClick={sendMessage} disabled={!input.trim() || loading}
              className="bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 text-black font-semibold px-5 py-2.5 rounded-xl text-sm transition-colors">
              Send
            </button>
          </div>
        </>
      )}
    </div>
  )
}

// ======================================================
// MAIN APP
// ======================================================
const TABS = [
  { id: 'hunt', label: 'Job Hunt', icon: <Search size={15} /> },
  { id: 'hr', label: 'HR Mode', icon: <Users size={15} /> },
  { id: 'chat', label: 'Resume Chat', icon: <MessageCircle size={15} /> },
]

function App() {
  const [activeTab, setActiveTab] = useState('hunt')
  const [step, setStep] = useState(() => Number(localStorage.getItem('step')) || 0)
  const [candidateId, setCandidateId] = useState(() => localStorage.getItem('candidateId') || '')
  const [candidateName, setCandidateName] = useState(() => localStorage.getItem('candidateName') || '')
  const [confirmedRoles, setConfirmedRoles] = useState<string[]>(() => JSON.parse(localStorage.getItem('confirmedRoles') || '[]'))
  const [searchLocation, setSearchLocation] = useState(() => localStorage.getItem('searchLocation') || 'Pune, India')
  const [searchCities, setSearchCities] = useState<string[]>(() => JSON.parse(localStorage.getItem('searchCities') || '[]'))
  const [searchWorkType, setSearchWorkType] = useState(() => localStorage.getItem('searchWorkType') || 'Hybrid')
  const [applyingJob, setApplyingJob] = useState<CrawledJob | null>(null)
  
  const [profile, setProfile] = useState<KnowledgeBase>(() => {
    const saved = localStorage.getItem('profile')
    return saved ? JSON.parse(saved) : {
      name: '', email: '', skills: [], experience_years: 0,
      phone: '', expected_salary: '', current_ctc: '', notice_period: ''
    }
  })

  // Lifted Jobs State (not persisted to localStorage as it's too large/ephemeral)
  const [jobs, setJobs] = useState<CrawledJob[]>([])
  const [jobsLoading, setJobsLoading] = useState(false)
  const [jobsError, setJobsError] = useState('')

  // Persistence
  useEffect(() => {
    localStorage.setItem('step', step.toString())
    localStorage.setItem('candidateId', candidateId)
    localStorage.setItem('candidateName', candidateName)
    localStorage.setItem('confirmedRoles', JSON.stringify(confirmedRoles))
    localStorage.setItem('searchLocation', searchLocation)
    localStorage.setItem('searchCities', JSON.stringify(searchCities))
    localStorage.setItem('searchWorkType', searchWorkType)
    localStorage.setItem('profile', JSON.stringify(profile))
  }, [step, candidateId, candidateName, confirmedRoles, searchLocation, searchCities, searchWorkType, profile])

  // Clear jobs cache if search criteria change, so user gets fresh results on next mount
  useEffect(() => {
    if (jobs.length > 0) {
      setJobs([])
      setJobsError('')
    }
  }, [JSON.stringify(confirmedRoles), searchLocation, JSON.stringify(searchCities), searchWorkType])

  return (
    <div className="min-h-screen bg-zinc-950 text-white" style={{ fontFamily: 'Inter, sans-serif' }}>
      {/* Header */}
      <header className="border-b border-zinc-800/60 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-emerald-500 rounded-lg flex items-center justify-center">
            <Briefcase size={14} className="text-black" />
          </div>
          <span className="font-bold text-white tracking-tight">Unthinkable</span>
          <span className="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full ml-1">v2</span>
        </div>
        {/* Tabs */}
        <div className="flex bg-zinc-900 border border-zinc-800 rounded-xl p-1 gap-1">
          {TABS.map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all
                ${activeTab === t.id ? 'bg-emerald-500 text-black' : 'text-zinc-400 hover:text-white'}`}>
              {t.icon} <span className="hidden sm:block">{t.label}</span>
            </button>
          ))}
        </div>
        {candidateName && (
          <div className="flex items-center gap-4">
            <span className="text-sm text-zinc-400 hidden md:block">
              <span className="text-white font-medium">{candidateName}</span>
            </span>
            <button onClick={() => {
              localStorage.clear();
              window.location.reload();
            }} className="p-2 hover:bg-zinc-800 rounded-lg text-zinc-500 hover:text-red-400 transition-colors" title="Logout">
              <LogOut size={16} />
            </button>
          </div>
        )}
      </header>

      <main className="max-w-5xl mx-auto px-4 py-10">
        <AnimatePresence mode="wait">
          {!candidateId ? (
            <LoginScreen key="login" onLogin={(data) => {
              setCandidateId(data.candidate_id);
              setCandidateName(data.candidate_name);
              if (data.profile) setProfile(data.profile);
              if (data.has_resume) setStep(2); // Skip upload if they have a resume
              else setStep(0); // Proceed to upload if not
            }} />
          ) : (
            activeTab === 'hunt' ? (
              <motion.div key="hunt" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                {step < 4 && <Stepper step={step} />}
                <AnimatePresence mode="wait">
                  {step === 0 && (
                    <motion.div key="s0" exit={{ opacity: 0, y: -10 }}>
                      <div className="text-center mb-10">
                        <h1 className="text-4xl font-bold tracking-tight mb-3">
                          Upload your{' '}
                          <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-400">
                            resume.
                          </span>
                        </h1>
                        <p className="text-zinc-400 max-w-md mx-auto">
                          AI picks your best roles, crawls jobs globally, and prepares every application.
                        </p>
                      </div>
                      <UploadStep candidateId={candidateId} onUploaded={(id, name, data) => { 
                        setCandidateId(id); 
                        setCandidateName(name); 
                        if (data?.knowledge_base) setProfile(data.knowledge_base);
                        setStep(1);
                      }} />
                    </motion.div>
                  )}
                  {step === 1 && (
                    <motion.div key="s1" exit={{ opacity: 0, y: -10 }}>
                      <OnboardingStep candidateId={candidateId} profile={profile} onComplete={(updated) => {
                        setProfile(updated);
                        setStep(2);
                      }} />
                    </motion.div>
                  )}
                  {step === 2 && (
                    <motion.div key="s2" exit={{ opacity: 0, y: -10 }}>
                      <RoleSelectionStep candidateId={candidateId} candidateName={candidateName}
                        onConfirm={(roles, loc, cities, wt) => {
                          setConfirmedRoles(roles); setSearchLocation(loc); setSearchCities(cities); setSearchWorkType(wt); setStep(3)
                        }} />
                    </motion.div>
                  )}
                  {step === 3 && (
                    <motion.div key="s3" exit={{ opacity: 0, y: -10 }}>
                      <div className="flex items-center gap-2 mb-6">
                        <button type="button" onClick={() => setStep(2)} className="text-zinc-500 hover:text-white text-sm transition-colors">← Back</button>
                      </div>
                      <JobFeedStep roles={confirmedRoles}
                        location={searchLocation} cities={searchCities} workType={searchWorkType}
                        jobs={jobs} loading={jobsLoading} error={jobsError}
                        onSearch={() => {
                          setJobsLoading(true)
                          setJobsError('')
                          fetch(`${API}/api/v2/discover-jobs/`, {
                            method: 'POST', headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ candidate_id: candidateId, user_selected_roles: confirmedRoles, location: searchLocation, cities: searchCities, work_type: searchWorkType })
                          }).then(r => r.json()).then(data => setJobs(data.crawled_jobs || []))
                            .catch(e => setJobsError(e.message)).finally(() => setJobsLoading(false))
                        }}
                        onApply={(job) => { setApplyingJob(job); setStep(4) }} />
                    </motion.div>
                  )}
                  {step === 4 && (
                    <motion.div key="s4" exit={{ opacity: 0, y: -10 }}>
                      <button type="button" onClick={() => { setApplyingJob(null); setStep(3) }} className="text-zinc-500 hover:text-white text-sm transition-colors mb-4 block">← Back to jobs</button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ) : activeTab === 'hr' ? (
              <motion.div key="hr" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <HRModeTab candidateId={candidateId} />
              </motion.div>
            ) : (
              <motion.div key="chat" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ResumeChatTab candidateId={candidateId} />
              </motion.div>
            )
          )}
        </AnimatePresence>
      </main>

      {/* Apply Modal */}
      <AnimatePresence>
        {applyingJob && step === 4 && (
          <ApplyModal candidateId={candidateId} job={applyingJob}
            onClose={() => { setApplyingJob(null); setStep(3) }}
            onIdSync={(id) => setCandidateId(id)} />
        )}
      </AnimatePresence>
    </div>
  )
}

export default function AppWrapper() {
  // IMPORTANT: Replace this with your own Google Client ID from Google Cloud Console
  // You can also stick it in a .env file later as VITE_GOOGLE_CLIENT_ID
  const GOOGLE_CLIENT_ID = "668900299069-skpi9fp6k39ao7leh4ojv637gqkun4id.apps.googleusercontent.com" 

  if (GOOGLE_CLIENT_ID.includes("PLACEHOLDER") || GOOGLE_CLIENT_ID.startsWith("633215682898") || GOOGLE_CLIENT_ID.startsWith("your_google")) {
    console.warn("Unthinkable: Google Client ID is still the placeholder. Login will fail with 'invalid_client'.")
  }

  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <App />
    </GoogleOAuthProvider>
  )
}