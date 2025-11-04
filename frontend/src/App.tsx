import React, { useState } from 'react';
import { Upload, Search, MessageSquare, Users, CheckCircle, XCircle, Sparkles, FileText, Briefcase } from 'lucide-react';

const API_URL = 'http://localhost:8000';

interface MatchResult {
  candidate_id: string;
  candidate_name: string;
  match_score: number;
  justification: string;
  matched_skills: string[];
  missing_skills: string[];
  experience_match: boolean;
}

interface Candidate {
  candidate_id: string;
  candidate_name: string;
}

// Utility function to convert markdown-style text to formatted HTML
const formatText = (text: string) => {
  if (!text) return text;
  
  // Convert **bold** to <strong>
  let formatted = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  
  // Convert single * bullets to proper bullets
  formatted = formatted.replace(/^\* /gm, '• ');
  
  // Convert *italic* to <em>
  formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  
  return formatted;
};

// Component to render formatted text
const FormattedText: React.FC<{ text: string; className?: string }> = ({ text, className = '' }) => {
  const formatted = formatText(text);
  return <div className={className} dangerouslySetInnerHTML={{ __html: formatted }} />;
};

export default function ResumeScreener() {
  const [activeTab, setActiveTab] = useState<'upload' | 'match' | 'chat'>('upload');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [jobTitle, setJobTitle] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [requiredSkills, setRequiredSkills] = useState('');
  const [experienceYears, setExperienceYears] = useState('');
  const [matchResults, setMatchResults] = useState<MatchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState('');
  const [chatQuestion, setChatQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState<Array<{q: string, a: string}>>([]);

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadedFile) return;

    setLoading(true);
    setUploadStatus('Uploading...');

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch(`${API_URL}/upload-resume/`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setUploadStatus(`✓ Resume uploaded successfully for ${data.data.name}`);
      setUploadedFile(null);
      
      // Refresh candidates list
      fetchCandidates();
    } catch (error) {
      setUploadStatus('✗ Upload failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchCandidates = async () => {
    try {
      const response = await fetch(`${API_URL}/candidates/`);
      const data = await response.json();
      setCandidates(data.candidates);
    } catch (error) {
      console.error('Failed to fetch candidates');
    }
  };

  const handleJobMatch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    const skillsArray = requiredSkills.split(',').map(s => s.trim());

    try {
      const response = await fetch(`${API_URL}/match-job/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: jobTitle,
          description: jobDescription,
          required_skills: skillsArray,
          experience_years: parseInt(experienceYears) || 0,
        }),
      });

      const data = await response.json();
      setMatchResults(data.matches);
    } catch (error) {
      console.error('Matching failed');
    } finally {
      setLoading(false);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCandidate || !chatQuestion.trim()) return;

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat-resume/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidate_id: selectedCandidate,
          question: chatQuestion,
        }),
      });

      const data = await response.json();
      setChatHistory([...chatHistory, { q: chatQuestion, a: data.answer }]);
      setChatQuestion('');
    } catch (error) {
      console.error('Chat failed');
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    fetchCandidates();
  }, []);

  const getScoreColor = (score: number) => {
    if (score >= 8) return 'text-green-600 bg-green-50';
    if (score >= 6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  Smart Resume Screener
                </h1>
                <p className="text-sm text-gray-600 mt-1">AI-Powered Candidate Matching</p>
              </div>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-indigo-50 rounded-lg">
              <Users className="w-5 h-5 text-indigo-600" />
              <span className="text-sm font-medium text-indigo-900">{candidates.length} Candidates</span>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex gap-2 bg-white p-2 rounded-xl shadow-sm border border-gray-100">
          {[
            { id: 'upload', label: 'Upload Resumes', icon: Upload },
            { id: 'match', label: 'Match Job', icon: Search },
            { id: 'chat', label: 'Chat with Resume', icon: MessageSquare },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                activeTab === id
                  ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg shadow-indigo-200'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <Icon className="w-5 h-5" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 pb-12">
        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8">
            <div className="max-w-2xl mx-auto">
              <div className="text-center mb-8">
                <FileText className="w-16 h-16 text-indigo-600 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Candidate Resumes</h2>
                <p className="text-gray-600">Upload PDF resumes to analyze and match with job descriptions</p>
              </div>

              <form onSubmit={handleFileUpload} className="space-y-6">
                <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-indigo-400 transition-colors">
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <label className="cursor-pointer">
                    <span className="text-indigo-600 font-medium hover:text-indigo-700">Choose a file</span>
                    <span className="text-gray-600"> or drag and drop</span>
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(e) => setUploadedFile(e.target.files?.[0] || null)}
                      className="hidden"
                    />
                  </label>
                  {uploadedFile && (
                    <p className="mt-4 text-sm text-gray-700 font-medium">{uploadedFile.name}</p>
                  )}
                </div>

                <button
                  type="submit"
                  disabled={!uploadedFile || loading}
                  className="w-full py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-indigo-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Processing...' : 'Upload Resume'}
                </button>

                {uploadStatus && (
                  <div className={`p-4 rounded-lg ${uploadStatus.includes('✓') ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'}`}>
                    {uploadStatus}
                  </div>
                )}
              </form>
            </div>
          </div>
        )}

        {/* Match Tab */}
        {activeTab === 'match' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8">
              <div className="flex items-center gap-3 mb-6">
                <Briefcase className="w-8 h-8 text-indigo-600" />
                <h2 className="text-2xl font-bold text-gray-900">Job Description</h2>
              </div>

              <form onSubmit={handleJobMatch} className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Job Title</label>
                  <input
                    type="text"
                    value={jobTitle}
                    onChange={(e) => setJobTitle(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    placeholder="e.g., Senior Software Engineer"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Job Description</label>
                  <textarea
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent h-32"
                    placeholder="Describe the role, responsibilities, and requirements..."
                  />
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Required Skills (comma-separated)</label>
                    <input
                      type="text"
                      value={requiredSkills}
                      onChange={(e) => setRequiredSkills(e.target.value)}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                      placeholder="Python, React, AWS, etc."
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Experience (years)</label>
                    <input
                      type="number"
                      value={experienceYears}
                      onChange={(e) => setExperienceYears(e.target.value)}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                      placeholder="5"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-indigo-200 transition-all disabled:opacity-50"
                >
                  {loading ? 'Analyzing...' : 'Find Matching Candidates'}
                </button>
              </form>
            </div>

            {/* Results */}
            {matchResults.length > 0 && (
              <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8">
                <h3 className="text-xl font-bold text-gray-900 mb-6">
                  Top Matches ({matchResults.length} candidates)
                </h3>

                <div className="space-y-4">
                  {matchResults.map((result) => (
                    <div
                      key={result.candidate_id}
                      className="border border-gray-200 rounded-xl p-6 hover:shadow-md transition-shadow"
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h4 className="text-lg font-semibold text-gray-900">{result.candidate_name}</h4>
                          <p className="text-sm text-gray-500">{result.candidate_id}</p>
                        </div>
                        <div className={`px-4 py-2 rounded-lg font-bold text-2xl ${getScoreColor(result.match_score)}`}>
                          {result.match_score.toFixed(1)}/10
                        </div>
                      </div>

                      <FormattedText text={result.justification} className="text-gray-700 mb-4" />

                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <CheckCircle className="w-5 h-5 text-green-600" />
                            <span className="font-medium text-gray-900">Matched Skills</span>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {result.matched_skills.map((skill, i) => (
                              <span key={i} className="px-3 py-1 bg-green-50 text-green-700 rounded-full text-sm">
                                {skill}
                              </span>
                            ))}
                          </div>
                        </div>

                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <XCircle className="w-5 h-5 text-red-600" />
                            <span className="font-medium text-gray-900">Missing Skills</span>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {result.missing_skills.map((skill, i) => (
                              <span key={i} className="px-3 py-1 bg-red-50 text-red-700 rounded-full text-sm">
                                {skill}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Chat Tab */}
        {activeTab === 'chat' && (
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8">
            <div className="flex items-center gap-3 mb-6">
              <MessageSquare className="w-8 h-8 text-indigo-600" />
              <h2 className="text-2xl font-bold text-gray-900">Chat with Resume</h2>
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">Select Candidate</label>
              <select
                value={selectedCandidate}
                onChange={(e) => {
                  setSelectedCandidate(e.target.value);
                  setChatHistory([]);
                }}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              >
                <option value="">Choose a candidate...</option>
                {candidates.map((c) => (
                  <option key={c.candidate_id} value={c.candidate_id}>
                    {c.candidate_name}
                  </option>
                ))}
              </select>
            </div>

            {selectedCandidate && (
              <>
                <div className="bg-gray-50 rounded-xl p-6 mb-6 max-h-96 overflow-y-auto">
                  {chatHistory.length === 0 ? (
                    <p className="text-gray-500 text-center py-8">Ask a question about the resume...</p>
                  ) : (
                    <div className="space-y-4">
                      {chatHistory.map((chat, i) => (
                        <div key={i} className="space-y-2">
                          <div className="bg-indigo-100 rounded-lg p-4">
                            <p className="font-medium text-indigo-900">Q: {chat.q}</p>
                          </div>
                          <div className="bg-white border border-gray-200 rounded-lg p-4">
                            <FormattedText text={chat.a} className="text-gray-800" />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <form onSubmit={handleChatSubmit} className="flex gap-3">
                  <input
                    type="text"
                    value={chatQuestion}
                    onChange={(e) => setChatQuestion(e.target.value)}
                    className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    placeholder="Ask about skills, experience, education..."
                  />
                  <button
                    type="submit"
                    disabled={loading || !chatQuestion.trim()}
                    className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold rounded-lg hover:shadow-lg hover:shadow-indigo-200 transition-all disabled:opacity-50"
                  >
                    {loading ? 'Thinking...' : 'Ask'}
                  </button>
                </form>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}