import React, { useState, useEffect, useRef } from 'react';

const App = () => {
  const [query, setQuery] = useState('');
  const [sourceFilter, setSourceFilter] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [answer, setAnswer] = useState('');
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const answerRef = useRef(null);

  const validSources = ['clause', 'medical', 'case'];

  useEffect(() => {
    fetch('http://localhost:8080/query', { method: 'POST' })
      .then(res => res.json())
      .then(data => setSessionId(data.session_id))
      .catch(err => console.error('Failed to start a case session:', err));
  }, []);

  useEffect(() => {
    if (answerRef.current) {
      answerRef.current.scrollTop = answerRef.current.scrollHeight;
    }
  }, [answer]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setAnswer('');
    const requestBody = {
      query,
      source_filter: sourceFilter || null,
      session_id: sessionId
    };

    try {
      const response = await fetch('http://localhost:8080/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      if (response.headers.get('content-type').includes('text/event-stream')) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulatedAnswer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const token = line.slice(6);
              if (token) {
                accumulatedAnswer += token;
                setAnswer(accumulatedAnswer);
              }
            }
          }
        }
      } else {
        const data = await response.json();
        setAnswer(data.answer);
      }

      const historyRes = await fetch(`http://localhost:8080/history/${sessionId}`);
      const historyData = await historyRes.json();
      setHistory(historyData.history);
    } catch (err) {
      setAnswer('Error: could not retrieve a first-pass opinion.');
      console.error('Claims query failed:', err);
    } finally {
      setIsLoading(false);
    }

    setQuery('');
    setSourceFilter('');
  };

  const handleClearHistory = async () => {
    try {
      await fetch(`http://localhost:8080/history/${sessionId}`, { method: 'DELETE' });
      setHistory([]);
      setAnswer('');
      alert('Case history cleared');
    } catch (err) {
      console.error('Failed to clear case history:', err);
      alert('Failed to clear case history');
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-6">Insurance RAG Assistant</h1>
      <p className="text-center mb-4">Session: {sessionId || 'Loading...'}</p>

      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700">Claims question</label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              placeholder="Ask about a clause, bill, or similar case..."
              disabled={isLoading}
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700">Source filter (optional)</label>
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              disabled={isLoading}
            >
              <option value="">All sources</option>
              {validSources.map((source) => (
                <option key={source} value={source}>{source}</option>
              ))}
            </select>
          </div>
          <button
            type="submit"
            className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 disabled:bg-gray-400"
            disabled={isLoading}
          >
            {isLoading ? 'Reviewing...' : 'Submit'}
          </button>
        </form>
      </div>

      {answer && (
        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-xl font-semibold mb-2">First-pass opinion</h2>
          <div
            ref={answerRef}
            className="max-h-96 overflow-y-auto prose prose-sm"
            dangerouslySetInnerHTML={{ __html: answer.replace(/\n/g, '<br>') }}
          ></div>
        </div>
      )}

      {history.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Recent case history</h2>
            <button
              onClick={handleClearHistory}
              className="text-red-600 hover:text-red-800"
            >
              Clear history
            </button>
          </div>
          <div className="space-y-4">
            {history.map((entry, idx) => (
              <div key={idx}>
                <p className="font-medium">Q: {entry.question}</p>
                <p className="text-gray-600">A: {entry.answer}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
