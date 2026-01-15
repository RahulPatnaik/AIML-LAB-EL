const { useState, useEffect } = React;

const API_BASE_URL = 'http://localhost:8000';

// Main App Component with Routing
function App() {
    const [currentPage, setCurrentPage] = useState('home'); // 'home', 'predict', 'shap', 'lime', 'feature-space', 'decision-path', 'full-check'
    const [disorders, setDisorders] = useState([]);
    const [selectedDisorder, setSelectedDisorder] = useState('');
    const [patients, setPatients] = useState([]);
    const [selectedPatient, setSelectedPatient] = useState(null);
    const [sampleData, setSampleData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Fetch disorders, patients and sample on mount
    useEffect(() => {
        fetchDisorders();
        fetchPatients();
        fetchSample();
    }, []);

    const fetchDisorders = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/disorders`);
            const data = await response.json();
            setDisorders(data.disorders);
            if (data.disorders.length > 0) {
                setSelectedDisorder(data.disorders[0].safe_name);
            }
        } catch (err) {
            setError('Failed to fetch disorders');
        }
    };

    const fetchPatients = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/patients`);
            const data = await response.json();
            setPatients(data.patients);
            if (data.patients.length > 0) {
                setSelectedPatient(data.patients[0].id);
            }
        } catch (err) {
            setError('Failed to fetch patients');
        }
    };

    const fetchSample = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/sample`);
            const data = await response.json();
            setSampleData(data);
        } catch (err) {
            setError('Failed to fetch sample data');
        }
    };

    const navigateTo = (page) => {
        setCurrentPage(page);
        setError(null);
    };

    // Render different pages based on currentPage
    if (currentPage === 'home') {
        return (
            <HomePage
                disorders={disorders}
                selectedDisorder={selectedDisorder}
                setSelectedDisorder={setSelectedDisorder}
                patients={patients}
                selectedPatient={selectedPatient}
                setSelectedPatient={setSelectedPatient}
                sampleData={sampleData}
                navigateTo={navigateTo}
                error={error}
            />
        );
    }

    // Other pages with navigation
    return (
        <div className="app">
            <HomeButton onClick={() => navigateTo('home')} />

            {currentPage === 'predict' && (
                <PredictPage
                    disorder={selectedDisorder}
                    loading={loading}
                    setLoading={setLoading}
                    error={error}
                    setError={setError}
                />
            )}

            {currentPage === 'shap' && (
                <SHAPPage
                    disorder={selectedDisorder}
                    loading={loading}
                    setLoading={setLoading}
                    error={error}
                    setError={setError}
                />
            )}

            {currentPage === 'lime' && (
                <LIMEPage
                    disorder={selectedDisorder}
                    loading={loading}
                    setLoading={setLoading}
                    error={error}
                    setError={setError}
                />
            )}

            {currentPage === 'feature-space' && (
                <FeatureSpacePage
                    disorder={selectedDisorder}
                    loading={loading}
                    setLoading={setLoading}
                    error={error}
                    setError={setError}
                />
            )}

            {currentPage === 'decision-path' && (
                <DecisionPathPage
                    disorder={selectedDisorder}
                    loading={loading}
                    setLoading={setLoading}
                    error={error}
                    setError={setError}
                />
            )}

            {currentPage === 'tsne' && (
                <TSNEPage
                    disorder={selectedDisorder}
                    loading={loading}
                    setLoading={setLoading}
                    error={error}
                    setError={setError}
                />
            )}

            {currentPage === 'umap' && (
                <UMAPPage
                    disorder={selectedDisorder}
                    loading={loading}
                    setLoading={setLoading}
                    error={error}
                    setError={setError}
                />
            )}

            {currentPage === 'full-check' && (
                <FullCheckPage
                    patientId={selectedPatient}
                    loading={loading}
                    setLoading={setLoading}
                    error={error}
                    setError={setError}
                />
            )}
        </div>
    );
}

// Home Button Component
function HomeButton({ onClick }) {
    return (
        <button className="home-button" onClick={onClick}>
            ← HOME
        </button>
    );
}

// AI Explanation Card Component
function AIExplanationCard({ explanation, loading }) {
    if (loading) {
        return (
            <div className="ai-explanation-card">
                <h3>🤖 AI CLINICAL INTERPRETATION</h3>
                <div className="ai-loading">Generating explanation...</div>
            </div>
        );
    }

    if (!explanation || explanation.error) {
        return null; // Don't show if error or not available
    }

    return (
        <div className="ai-explanation-card">
            <h3>🤖 AI CLINICAL INTERPRETATION</h3>
            <div className="ai-explanation-text">
                {explanation.explanation}
            </div>
            <div className="ai-disclaimer">
                <small>⚠️ AI-generated interpretation for clinical decision support. Always validate with professional psychiatric assessment.</small>
            </div>
        </div>
    );
}

// HomePage Component
function HomePage({ disorders, selectedDisorder, setSelectedDisorder, patients, selectedPatient, setSelectedPatient, sampleData, navigateTo, error }) {
    const currentPatient = patients.find(p => p.id === selectedPatient);

    return (
        <div className="app">
            <header className="header">
                <h1>EEG MODEL EXPLAINABILITY</h1>
                <p>Psychiatric Disorder Classification Dashboard</p>
            </header>

            <div className="controls">
                <div className="control-group">
                    <label>SELECT PATIENT:</label>
                    <select
                        value={selectedPatient || ''}
                        onChange={(e) => setSelectedPatient(parseInt(e.target.value))}
                        style={{ fontSize: '14px', padding: '12px' }}
                    >
                        {patients.map(p => (
                            <option key={p.id} value={p.id}>
                                {p.label}
                            </option>
                        ))}
                    </select>
                </div>

                {currentPatient && (
                    <div className="sample-info" style={{ marginTop: '20px' }}>
                        <h3>SELECTED PATIENT</h3>
                        <div className="info-grid">
                            <div className="info-item">
                                <div className="info-label">ID</div>
                                <div className="info-value">#{currentPatient.id}</div>
                            </div>
                            <div className="info-item">
                                <div className="info-label">SEX</div>
                                <div className="info-value">{currentPatient.sex}</div>
                            </div>
                            <div className="info-item">
                                <div className="info-label">AGE</div>
                                <div className="info-value">{currentPatient.age}</div>
                            </div>
                            <div className="info-item">
                                <div className="info-label">ACTUAL DIAGNOSIS</div>
                                <div className="info-value">{currentPatient.specific_disorder}</div>
                            </div>
                        </div>
                    </div>
                )}

                <div className="control-group" style={{ marginTop: '30px' }}>
                    <label>SELECT DISORDER TO ANALYZE:</label>
                    <select
                        value={selectedDisorder}
                        onChange={(e) => setSelectedDisorder(e.target.value)}
                    >
                        {disorders.map(d => (
                            <option key={d.safe_name} value={d.safe_name}>
                                {d.name}
                            </option>
                        ))}
                    </select>
                </div>

                {error && <div className="error">ERROR: {error}</div>}

                <div className="button-grid" style={{ marginTop: '30px' }}>
                    <button className="nav-btn btn-danger" onClick={() => navigateTo('full-check')} style={{ gridColumn: '1 / -1', fontSize: '18px', padding: '20px' }}>
                        🔬 FULL CHECK - Analyze ALL Disorders
                    </button>
                </div>

                <div className="button-grid">
                    <button className="nav-btn btn-primary" onClick={() => navigateTo('predict')}>
                        🎯 PREDICT
                    </button>
                    <button className="nav-btn btn-secondary" onClick={() => navigateTo('shap')}>
                        📊 SHAP EXPLANATION
                    </button>
                    <button className="nav-btn btn-success" onClick={() => navigateTo('lime')}>
                        🔍 LIME EXPLANATION
                    </button>
                    <button className="nav-btn btn-info" onClick={() => navigateTo('feature-space')}>
                        🗺️ FEATURE SPACE (PCA)
                    </button>
                    <button className="nav-btn btn-warning" onClick={() => navigateTo('tsne')}>
                        🔬 t-SNE PROJECTION
                    </button>
                    <button className="nav-btn btn-warning" onClick={() => navigateTo('umap')}>
                        🌐 UMAP PROJECTION
                    </button>
                    <button className="nav-btn btn-info" onClick={() => navigateTo('decision-path')}>
                        🌳 DECISION PATH
                    </button>
                </div>
            </div>
        </div>
    );
}

// Sample Info Component
function SampleInfo({ data }) {
    return (
        <div className="sample-info">
            <h3>SAMPLE PATIENT</h3>
            <div className="info-grid">
                {data.metadata.no && (
                    <div className="info-item">
                        <div className="info-label">ID</div>
                        <div className="info-value">#{data.metadata.no}</div>
                    </div>
                )}
                {data.metadata.sex && (
                    <div className="info-item">
                        <div className="info-label">SEX</div>
                        <div className="info-value">{data.metadata.sex}</div>
                    </div>
                )}
                {data.metadata.age && (
                    <div className="info-item">
                        <div className="info-label">AGE</div>
                        <div className="info-value">{data.metadata.age}</div>
                    </div>
                )}
                <div className="info-item">
                    <div className="info-label">FEATURES</div>
                    <div className="info-value">{data.num_features}</div>
                </div>
            </div>
        </div>
    );
}

// PredictPage Component
function PredictPage({ disorder, loading, setLoading, error, setError }) {
    const [prediction, setPrediction] = useState(null);

    useEffect(() => {
        handlePredict();
    }, []);

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: disorder })
            });
            const data = await response.json();
            console.log('Predict Response:', data);
            setPrediction(data);
        } catch (err) {
            console.error('Predict Error:', err);
            setError('Prediction failed: ' + err.message);
        }
        setLoading(false);
    };

    return (
        <div className="page-container">
            <h1 className="page-title">PREDICTION RESULTS</h1>
            <div className="page-subtitle">{disorder.replace(/_/g, ' ')}</div>

            {loading && <div className="loading">PROCESSING...</div>}
            {error && <div className="error">ERROR: {error}</div>}

            {prediction && <PredictionCard data={prediction} />}
            {prediction && <AIExplanationCard explanation={prediction.ai_explanation} loading={loading} />}
        </div>
    );
}

// PredictionCard Component
function PredictionCard({ data }) {
    const pred = data.prediction || data;
    const predictionLabel = pred.prediction === 1 ? 'POSITIVE' : 'NEGATIVE';
    const badgeClass = pred.prediction === 1 ? 'badge-positive' : 'badge-negative';

    return (
        <div className="result-card">
            <div className={`prediction-badge ${badgeClass}`}>
                {predictionLabel}
            </div>

            <div className="metric">
                <span className="metric-label">PROBABILITY:</span>
                <span className="metric-value">{(pred.probability * 100).toFixed(2)}%</span>
            </div>

            <div className="metric">
                <span className="metric-label">THRESHOLD:</span>
                <span className="metric-value">{(pred.threshold * 100).toFixed(2)}%</span>
            </div>

            <div className="metric">
                <span className="metric-label">CONFIDENCE:</span>
                <span className={`metric-value confidence-${pred.confidence ? pred.confidence.toLowerCase() : 'medium'}`}>
                    {pred.confidence || 'N/A'}
                </span>
            </div>

            {pred.individual_predictions && (
                <>
                    <h3>INDIVIDUAL MODELS</h3>
                    <div className="metric">
                        <span className="metric-label">KNN:</span>
                        <span className="metric-value">{(pred.individual_predictions.knn * 100).toFixed(2)}%</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">RANDOM FOREST:</span>
                        <span className="metric-value">{(pred.individual_predictions.random_forest * 100).toFixed(2)}%</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">EXTRA TREES:</span>
                        <span className="metric-value">{(pred.individual_predictions.extra_trees * 100).toFixed(2)}%</span>
                    </div>
                </>
            )}
        </div>
    );
}

// SHAPPage Component
function SHAPPage({ disorder, loading, setLoading, error, setError }) {
    const [shapData, setShapData] = useState(null);

    useEffect(() => {
        handleSHAP();
    }, []);

    const handleSHAP = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/explain/shap`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: disorder })
            });
            const data = await response.json();
            console.log('SHAP Response:', data);
            setShapData(data);
        } catch (err) {
            console.error('SHAP Error:', err);
            setError('SHAP explanation failed: ' + err.message);
        }
        setLoading(false);
    };

    useEffect(() => {
        if (shapData && shapData.shap_values && !shapData.shap_values.error) {
            renderSHAPChart(shapData.shap_values);
        }
    }, [shapData]);

    const renderSHAPChart = (shap) => {
        if (!shap.top_features) return;

        setTimeout(() => {
            const chartDiv = document.getElementById('shap-chart');
            if (!chartDiv) return;

            const features = shap.top_features.slice(0, 15);
            const trace = {
                type: 'bar',
                x: features.map(f => f.shap_value),
                y: features.map(f => f.feature),
                orientation: 'h',
                marker: {
                    color: features.map(f => f.shap_value > 0 ? '#00ff00' : '#ff0000'),
                }
            };

            const layout = {
                title: { text: 'SHAP Feature Importance', font: { color: '#fff', size: 20 } },
                xaxis: { title: 'SHAP Value', gridcolor: '#333', color: '#fff' },
                yaxis: { title: '', color: '#fff' },
                height: 600,
                margin: { l: 250, r: 50, t: 80, b: 50 },
                paper_bgcolor: '#000',
                plot_bgcolor: '#111',
                font: { color: '#fff' }
            };

            Plotly.newPlot('shap-chart', [trace], layout);
        }, 100);
    };

    return (
        <div className="page-container">
            <h1 className="page-title">SHAP EXPLANATION</h1>
            <div className="page-subtitle">{disorder.replace(/_/g, ' ')}</div>

            {loading && <div className="loading">COMPUTING SHAP VALUES...</div>}
            {error && <div className="error">ERROR: {error}</div>}

            {shapData && shapData.shap_values && (
                <SHAPCard data={shapData} />
            )}
            {shapData && <AIExplanationCard explanation={shapData.ai_explanation} loading={loading} />}
        </div>
    );
}

// SHAPCard Component
function SHAPCard({ data }) {
    const shap = data.shap_values || data;

    if (shap.error) {
        return <div className="error">ERROR: {shap.error}</div>;
    }

    return (
        <div className="result-card">
            {shap.method && <p className="method-label">METHOD: {shap.method}</p>}

            {shap.base_value !== undefined && (
                <div className="metric">
                    <span className="metric-label">BASE VALUE:</span>
                    <span className="metric-value">{shap.base_value.toFixed(4)}</span>
                </div>
            )}

            {shap.prediction !== undefined && (
                <div className="metric">
                    <span className="metric-label">PREDICTION:</span>
                    <span className="metric-value">{shap.prediction.toFixed(4)}</span>
                </div>
            )}

            {shap.interpretation && (
                <div className="interpretation">
                    <p>{shap.interpretation}</p>
                </div>
            )}

            <div id="shap-chart" className="chart-container"></div>

            {shap.top_features && shap.top_features.length > 0 && (
                <>
                    <h3>TOP FEATURES</h3>
                    <div className="feature-list">
                        {shap.top_features.slice(0, 10).map((f, i) => (
                            <div key={i} className="feature-item">
                                <div className="feature-name">
                                    {f.feature}
                                    <span className="feature-importance">
                                        {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(4)}
                                    </span>
                                </div>
                                <div className="feature-value">
                                    Value: {f.feature_value.toFixed(4)} | Importance: {f.importance.toFixed(4)}
                                </div>
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
}

// LIMEPage Component
function LIMEPage({ disorder, loading, setLoading, error, setError }) {
    const [limeData, setLimeData] = useState(null);

    useEffect(() => {
        handleLIME();
    }, []);

    const handleLIME = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/explain/lime`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: disorder })
            });
            const data = await response.json();
            console.log('LIME Response:', data);
            setLimeData(data);
        } catch (err) {
            console.error('LIME Error:', err);
            setError('LIME explanation failed: ' + err.message);
        }
        setLoading(false);
    };

    useEffect(() => {
        if (limeData && limeData.lime_values && !limeData.lime_values.error) {
            renderLIMEChart(limeData.lime_values);
        }
    }, [limeData]);

    const renderLIMEChart = (lime) => {
        if (!lime.top_features) return;

        setTimeout(() => {
            const chartDiv = document.getElementById('lime-chart');
            if (!chartDiv) return;

            const features = lime.top_features.slice(0, 15);
            const trace = {
                type: 'bar',
                x: features.map(f => f.importance),
                y: features.map(f => f.feature || f.description),
                orientation: 'h',
                marker: {
                    color: features.map(f => f.importance > 0 ? '#00ffff' : '#ff00ff'),
                }
            };

            const layout = {
                title: { text: 'LIME Feature Importance', font: { color: '#fff', size: 20 } },
                xaxis: { title: 'LIME Weight', gridcolor: '#333', color: '#fff' },
                yaxis: { title: '', color: '#fff' },
                height: 600,
                margin: { l: 250, r: 50, t: 80, b: 50 },
                paper_bgcolor: '#000',
                plot_bgcolor: '#111',
                font: { color: '#fff' }
            };

            Plotly.newPlot('lime-chart', [trace], layout);
        }, 100);
    };

    return (
        <div className="page-container">
            <h1 className="page-title">LIME EXPLANATION</h1>
            <div className="page-subtitle">{disorder.replace(/_/g, ' ')}</div>
            <div className="warning-message">⚠️ LIME computation may take 30-60 seconds</div>

            {loading && <div className="loading">COMPUTING LIME VALUES (30-60s)...</div>}
            {error && <div className="error">ERROR: {error}</div>}

            {limeData && limeData.lime_values && (
                <LIMECard data={limeData} />
            )}
            {limeData && <AIExplanationCard explanation={limeData.ai_explanation} loading={loading} />}
        </div>
    );
}

// LIMECard Component
function LIMECard({ data }) {
    const lime = data.lime_values || data;

    if (lime.error) {
        return <div className="error">ERROR: {lime.error}</div>;
    }

    return (
        <div className="result-card">
            {lime.method && <p className="method-label">METHOD: {lime.method}</p>}

            {lime.prediction_probability !== undefined && (
                <div className="metric">
                    <span className="metric-label">PREDICTION PROBABILITY:</span>
                    <span className="metric-value">{(lime.prediction_probability * 100).toFixed(2)}%</span>
                </div>
            )}

            {lime.interpretation && (
                <div className="interpretation">
                    <p>{lime.interpretation}</p>
                </div>
            )}

            <div id="lime-chart" className="chart-container"></div>

            {lime.top_features && lime.top_features.length > 0 && (
                <>
                    <h3>TOP FEATURES</h3>
                    <div className="feature-list">
                        {lime.top_features.slice(0, 10).map((f, i) => (
                            <div key={i} className="feature-item">
                                <div className="feature-name">
                                    {f.description || f.feature}
                                    <span className="feature-importance">
                                        {f.importance > 0 ? '+' : ''}{f.importance.toFixed(4)}
                                    </span>
                                </div>
                                <div className="feature-value">
                                    Value: {f.feature_value.toFixed(4)}
                                </div>
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
}

// FeatureSpacePage Component
function FeatureSpacePage({ disorder, loading, setLoading, error, setError }) {
    const [featureSpace, setFeatureSpace] = useState(null);

    useEffect(() => {
        handleFeatureSpace();
    }, []);

    const handleFeatureSpace = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/visualize/feature_space`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: disorder })
            });
            const data = await response.json();
            console.log('Feature Space Response:', data);
            setFeatureSpace(data);
        } catch (err) {
            console.error('Feature Space Error:', err);
            setError('Feature space visualization failed: ' + err.message);
        }
        setLoading(false);
    };

    useEffect(() => {
        if (featureSpace && featureSpace.visualization && !featureSpace.visualization.error) {
            renderFeatureSpaceChart(featureSpace.visualization);
        }
    }, [featureSpace]);

    const renderFeatureSpaceChart = (feat) => {
        if (!feat.coordinates) return;

        setTimeout(() => {
            const chartDiv = document.getElementById('pca-chart');
            if (!chartDiv) return;

            const trace = {
                type: 'scatter',
                mode: 'markers',
                x: [feat.coordinates.pc1],
                y: [feat.coordinates.pc2],
                marker: {
                    size: 20,
                    color: '#00ff00',
                    symbol: 'star',
                    line: { color: '#fff', width: 2 }
                },
                text: ['Sample Patient'],
                hoverinfo: 'text'
            };

            const layout = {
                title: { text: 'PCA Feature Space', font: { color: '#fff', size: 20 } },
                xaxis: {
                    title: `PC1 (${feat.explained_variance ? (feat.explained_variance.pc1 * 100).toFixed(1) : '?'}% variance)`,
                    zeroline: true,
                    gridcolor: '#333',
                    color: '#fff'
                },
                yaxis: {
                    title: `PC2 (${feat.explained_variance ? (feat.explained_variance.pc2 * 100).toFixed(1) : '?'}% variance)`,
                    zeroline: true,
                    gridcolor: '#333',
                    color: '#fff'
                },
                height: 600,
                paper_bgcolor: '#000',
                plot_bgcolor: '#111',
                font: { color: '#fff' }
            };

            Plotly.newPlot('pca-chart', [trace], layout);
        }, 100);
    };

    return (
        <div className="page-container">
            <h1 className="page-title">FEATURE SPACE</h1>
            <div className="page-subtitle">{disorder.replace(/_/g, ' ')}</div>

            {loading && <div className="loading">COMPUTING PCA...</div>}
            {error && <div className="error">ERROR: {error}</div>}

            {featureSpace && featureSpace.visualization && (
                <FeatureSpaceCard data={featureSpace} />
            )}
            {featureSpace && <AIExplanationCard explanation={featureSpace.ai_explanation} loading={loading} />}
        </div>
    );
}

// FeatureSpaceCard Component
function FeatureSpaceCard({ data }) {
    const feat = data.visualization || data;

    if (feat.error) {
        return <div className="error">ERROR: {feat.error}</div>;
    }

    return (
        <div className="result-card">
            {feat.method && <p className="method-label">METHOD: {feat.method}</p>}

            {feat.coordinates && (
                <>
                    <div className="metric">
                        <span className="metric-label">PC1 COORDINATE:</span>
                        <span className="metric-value">{feat.coordinates.pc1.toFixed(4)}</span>
                    </div>

                    <div className="metric">
                        <span className="metric-label">PC2 COORDINATE:</span>
                        <span className="metric-value">{feat.coordinates.pc2.toFixed(4)}</span>
                    </div>
                </>
            )}

            {feat.explained_variance && (
                <div className="metric">
                    <span className="metric-label">TOTAL VARIANCE:</span>
                    <span className="metric-value">{(feat.explained_variance.total * 100).toFixed(2)}%</span>
                </div>
            )}

            {feat.interpretation && (
                <div className="interpretation">
                    <p>{feat.interpretation}</p>
                </div>
            )}

            <div id="pca-chart" className="chart-container"></div>
        </div>
    );
}

// DecisionPathPage Component
function DecisionPathPage({ disorder, loading, setLoading, error, setError }) {
    const [decisionPath, setDecisionPath] = useState(null);

    useEffect(() => {
        handleDecisionPath();
    }, []);

    const handleDecisionPath = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/visualize/decision_path`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: disorder })
            });
            const data = await response.json();
            console.log('Decision Path Response:', data);
            setDecisionPath(data);
        } catch (err) {
            console.error('Decision Path Error:', err);
            setError('Decision path visualization failed: ' + err.message);
        }
        setLoading(false);
    };

    useEffect(() => {
        if (decisionPath && decisionPath.decision_path && !decisionPath.decision_path.error) {
            renderDecisionPathChart(decisionPath.decision_path);
        }
    }, [decisionPath]);

    const renderDecisionPathChart = (dec) => {
        if (!dec.random_forest || !dec.random_forest.top_features) return;

        setTimeout(() => {
            const chartDiv = document.getElementById('decision-chart');
            if (!chartDiv) return;

            const features = dec.random_forest.top_features.slice(0, 15);
            const trace = {
                type: 'bar',
                x: features.map(f => f.importance),
                y: features.map(f => f.name || `Feature ${f.index}`),
                orientation: 'h',
                marker: { color: '#ffff00' }
            };

            const layout = {
                title: { text: 'Random Forest Feature Importance', font: { color: '#fff', size: 20 } },
                xaxis: { title: 'Importance', gridcolor: '#333', color: '#fff' },
                yaxis: { title: '', color: '#fff' },
                height: 600,
                margin: { l: 250, r: 50, t: 80, b: 50 },
                paper_bgcolor: '#000',
                plot_bgcolor: '#111',
                font: { color: '#fff' }
            };

            Plotly.newPlot('decision-chart', [trace], layout);
        }, 100);
    };

    return (
        <div className="page-container">
            <h1 className="page-title">DECISION PATH</h1>
            <div className="page-subtitle">{disorder.replace(/_/g, ' ')}</div>

            {loading && <div className="loading">ANALYZING DECISION PATH...</div>}
            {error && <div className="error">ERROR: {error}</div>}

            {decisionPath && decisionPath.decision_path && (
                <DecisionPathCard data={decisionPath} />
            )}
            {decisionPath && <AIExplanationCard explanation={decisionPath.ai_explanation} loading={loading} />}
        </div>
    );
}

// DecisionPathCard Component
function DecisionPathCard({ data }) {
    const dec = data.decision_path || data;

    if (dec.error) {
        return <div className="error">ERROR: {dec.error}</div>;
    }

    return (
        <div className="result-card">
            {dec.method && <p className="method-label">METHOD: {dec.method}</p>}

            {dec.interpretation && (
                <div className="interpretation">
                    <p>{dec.interpretation}</p>
                </div>
            )}

            {dec.random_forest && (
                <>
                    <h3>RANDOM FOREST</h3>
                    <div className="metric">
                        <span className="metric-label">NUMBER OF TREES:</span>
                        <span className="metric-value">{dec.random_forest.num_trees}</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">PATH LENGTH:</span>
                        <span className="metric-value">{dec.random_forest.path_length} nodes</span>
                    </div>
                </>
            )}

            {dec.knn_neighbors && (
                <>
                    <h3>KNN NEIGHBORS</h3>
                    <div className="metric">
                        <span className="metric-label">K:</span>
                        <span className="metric-value">{dec.knn_neighbors.k}</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">DISTANCES:</span>
                        <span className="metric-value">
                            {dec.knn_neighbors.distances.slice(0, 5).map(d => d.toFixed(3)).join(', ')}
                        </span>
                    </div>
                </>
            )}

            <div id="decision-chart" className="chart-container"></div>
        </div>
    );
}

// t-SNE Page Component
function TSNEPage({ disorder, loading, setLoading, error, setError }) {
    const [tsneData, setTsneData] = useState(null);

    useEffect(() => {
        handleTSNE();
    }, []);

    const handleTSNE = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/visualize/tsne`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: disorder })
            });
            const data = await response.json();
            console.log('t-SNE Response:', data);
            setTsneData(data);
        } catch (err) {
            console.error('t-SNE Error:', err);
            setError('t-SNE visualization failed: ' + err.message);
        }
        setLoading(false);
    };

    return (
        <div className="page-container">
            <h1 className="page-title">t-SNE PROJECTION</h1>
            <div className="page-subtitle">{disorder.replace(/_/g, ' ')}</div>

            {loading && <div className="loading">COMPUTING t-SNE (MAY TAKE 10-30 SECONDS)...</div>}
            {error && <div className="error">ERROR: {error}</div>}

            {tsneData && tsneData.tsne && (
                <TSNECard data={tsneData} />
            )}
            {tsneData && <AIExplanationCard explanation={tsneData.ai_explanation} loading={loading} />}
        </div>
    );
}

// t-SNE Card Component
function TSNECard({ data }) {
    const tsne = data.tsne || data;

    useEffect(() => {
        if (!tsne || !tsne.coordinates) {
            console.log('TSNECard: No coordinates available');
            return;
        }

        console.log('TSNECard: Mounted, rendering chart...');
        console.log('TSNECard: Training samples:', tsne.training_context?.train_coords_1?.length || 0);

        // Wait a bit for DOM to be ready
        const timer = setTimeout(() => {
            const chartDiv = document.getElementById('tsne-chart');
            if (!chartDiv) {
                console.error('TSNECard: Chart div not found!');
                return;
            }

            console.log('TSNECard: Chart div found, creating plot...');

            const traces = [];

            // Training data context
            if (tsne.training_context && tsne.training_context.train_coords_1 &&
                tsne.training_context.train_coords_1.length > 0) {
                traces.push({
                    type: 'scatter',
                    mode: 'markers',
                    x: tsne.training_context.train_coords_1,
                    y: tsne.training_context.train_coords_2,
                    marker: { size: 8, color: '#333', opacity: 0.6 },
                    name: 'Training Samples',
                    hoverinfo: 'skip'
                });
            }

            // Test sample
            traces.push({
                type: 'scatter',
                mode: 'markers',
                x: [tsne.coordinates.dim1],
                y: [tsne.coordinates.dim2],
                marker: {
                    size: 20,
                    color: '#ff00ff',
                    symbol: 'star',
                    line: { color: '#fff', width: 2 }
                },
                name: 'Test Sample',
                text: ['Sample Patient'],
                hoverinfo: 'text'
            });

            const layout = {
                title: { text: 't-SNE Projection', font: { color: '#fff', size: 20 } },
                xaxis: {
                    title: 't-SNE Dimension 1',
                    zeroline: true,
                    gridcolor: '#333',
                    color: '#fff'
                },
                yaxis: {
                    title: 't-SNE Dimension 2',
                    zeroline: true,
                    gridcolor: '#333',
                    color: '#fff'
                },
                height: 600,
                paper_bgcolor: '#000',
                plot_bgcolor: '#111',
                font: { color: '#fff' },
                showlegend: true,
                legend: { font: { color: '#fff' } }
            };

            console.log('TSNECard: Calling Plotly.newPlot with', traces.length, 'traces');
            Plotly.newPlot('tsne-chart', traces, layout)
                .then(() => console.log('TSNECard: Plot created successfully!'))
                .catch(err => console.error('TSNECard: Plotly error:', err));
        }, 100);

        return () => clearTimeout(timer);
    }, [tsne]);

    if (tsne.error) {
        return <div className="error">ERROR: {tsne.error}</div>;
    }

    return (
        <div className="result-card">
            {tsne.method && <p className="method-label">METHOD: {tsne.method}</p>}

            {tsne.coordinates && (
                <>
                    <div className="metric">
                        <span className="metric-label">DIMENSION 1:</span>
                        <span className="metric-value">{tsne.coordinates.dim1.toFixed(4)}</span>
                    </div>

                    <div className="metric">
                        <span className="metric-label">DIMENSION 2:</span>
                        <span className="metric-value">{tsne.coordinates.dim2.toFixed(4)}</span>
                    </div>
                </>
            )}

            {tsne.training_context && (
                <div className="metric">
                    <span className="metric-label">TRAINING SAMPLES:</span>
                    <span className="metric-value">{tsne.training_context.num_samples}</span>
                </div>
            )}

            {tsne.training_context && tsne.training_context.nearest_neighbors && (
                <>
                    <h3>NEAREST NEIGHBORS IN t-SNE SPACE</h3>
                    <div className="metric">
                        <span className="metric-label">DISTANCES:</span>
                        <span className="metric-value">
                            {tsne.training_context.nearest_neighbors.distances.slice(0, 5).map(d => d.toFixed(3)).join(', ')}
                        </span>
                    </div>
                </>
            )}

            {tsne.interpretation && (
                <div className="interpretation">
                    <p>{tsne.interpretation}</p>
                </div>
            )}

            <div id="tsne-chart" className="chart-container"></div>
        </div>
    );
}

// UMAP Page Component
function UMAPPage({ disorder, loading, setLoading, error, setError }) {
    const [umapData, setUmapData] = useState(null);

    useEffect(() => {
        handleUMAP();
    }, []);

    const handleUMAP = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/visualize/umap`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: disorder })
            });
            const data = await response.json();
            console.log('UMAP Response:', data);
            setUmapData(data);
        } catch (err) {
            console.error('UMAP Error:', err);
            setError('UMAP visualization failed: ' + err.message);
        }
        setLoading(false);
    };

    return (
        <div className="page-container">
            <h1 className="page-title">UMAP PROJECTION</h1>
            <div className="page-subtitle">{disorder.replace(/_/g, ' ')}</div>

            {loading && <div className="loading">COMPUTING UMAP (MAY TAKE 10-30 SECONDS)...</div>}
            {error && <div className="error">ERROR: {error}</div>}

            {umapData && umapData.umap && (
                <UMAPCard data={umapData} />
            )}
            {umapData && <AIExplanationCard explanation={umapData.ai_explanation} loading={loading} />}
        </div>
    );
}

// UMAP Card Component
function UMAPCard({ data }) {
    const umap = data.umap || data;

    useEffect(() => {
        if (!umap || !umap.coordinates) {
            console.log('UMAPCard: No coordinates available');
            return;
        }

        console.log('UMAPCard: Mounted, rendering chart...');
        console.log('UMAPCard: Training samples:', umap.training_context?.train_coords_1?.length || 0);

        const timer = setTimeout(() => {
            const chartDiv = document.getElementById('umap-chart');
            if (!chartDiv) {
                console.error('UMAPCard: Chart div not found!');
                return;
            }

            console.log('UMAPCard: Chart div found, creating plot...');

            const traces = [];

            // Training data context
            if (umap.training_context && umap.training_context.train_coords_1 &&
                umap.training_context.train_coords_1.length > 0) {
                traces.push({
                    type: 'scatter',
                    mode: 'markers',
                    x: umap.training_context.train_coords_1,
                    y: umap.training_context.train_coords_2,
                    marker: { size: 8, color: '#333', opacity: 0.6 },
                    name: 'Training Samples',
                    hoverinfo: 'skip'
                });
            }

            // Test sample
            traces.push({
                type: 'scatter',
                mode: 'markers',
                x: [umap.coordinates.dim1],
                y: [umap.coordinates.dim2],
                marker: {
                    size: 20,
                    color: '#ff6600',
                    symbol: 'star',
                    line: { color: '#fff', width: 2 }
                },
                name: 'Test Sample',
                text: ['Sample Patient'],
                hoverinfo: 'text'
            });

            const layout = {
                title: { text: 'UMAP Projection', font: { color: '#fff', size: 20 } },
                xaxis: {
                    title: 'UMAP Dimension 1',
                    zeroline: true,
                    gridcolor: '#333',
                    color: '#fff'
                },
                yaxis: {
                    title: 'UMAP Dimension 2',
                    zeroline: true,
                    gridcolor: '#333',
                    color: '#fff'
                },
                height: 600,
                paper_bgcolor: '#000',
                plot_bgcolor: '#111',
                font: { color: '#fff' },
                showlegend: true,
                legend: { font: { color: '#fff' } }
            };

            console.log('UMAPCard: Calling Plotly.newPlot with', traces.length, 'traces');
            Plotly.newPlot('umap-chart', traces, layout)
                .then(() => console.log('UMAPCard: Plot created successfully!'))
                .catch(err => console.error('UMAPCard: Plotly error:', err));
        }, 100);

        return () => clearTimeout(timer);
    }, [umap]);

    if (umap.error) {
        return <div className="error">ERROR: {umap.error}</div>;
    }

    return (
        <div className="result-card">
            {umap.method && <p className="method-label">METHOD: {umap.method}</p>}

            {umap.coordinates && (
                <>
                    <div className="metric">
                        <span className="metric-label">DIMENSION 1:</span>
                        <span className="metric-value">{umap.coordinates.dim1.toFixed(4)}</span>
                    </div>

                    <div className="metric">
                        <span className="metric-label">DIMENSION 2:</span>
                        <span className="metric-value">{umap.coordinates.dim2.toFixed(4)}</span>
                    </div>
                </>
            )}

            {umap.training_context && (
                <div className="metric">
                    <span className="metric-label">TRAINING SAMPLES:</span>
                    <span className="metric-value">{umap.training_context.num_samples}</span>
                </div>
            )}

            {umap.training_context && umap.training_context.nearest_neighbors && (
                <>
                    <h3>NEAREST NEIGHBORS IN UMAP SPACE</h3>
                    <div className="metric">
                        <span className="metric-label">DISTANCES:</span>
                        <span className="metric-value">
                            {umap.training_context.nearest_neighbors.distances.slice(0, 5).map(d => d.toFixed(3)).join(', ')}
                        </span>
                    </div>
                </>
            )}

            {umap.interpretation && (
                <div className="interpretation">
                    <p>{umap.interpretation}</p>
                </div>
            )}

            <div id="umap-chart" className="chart-container"></div>
        </div>
    );
}

// Full Check Page Component
function FullCheckPage({ patientId, loading, setLoading, error, setError }) {
    const [fullCheckData, setFullCheckData] = useState(null);

    useEffect(() => {
        handleFullCheck();
    }, [patientId]);

    const handleFullCheck = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/predict/full_check`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ patient_id: patientId })
            });
            const data = await response.json();
            console.log('Full Check Response:', data);
            setFullCheckData(data);
        } catch (err) {
            console.error('Full Check Error:', err);
            setError('Full check failed: ' + err.message);
        }
        setLoading(false);
    };

    return (
        <div className="page-container">
            <h1 className="page-title">FULL DISORDER CHECK</h1>
            <div className="page-subtitle">Comprehensive Analysis Across All {fullCheckData?.total_checked || 17} Disorders</div>

            {loading && <div className="loading">RUNNING FULL CHECK (MAY TAKE 30-60 SECONDS)...</div>}
            {error && <div className="error">ERROR: {error}</div>}

            {fullCheckData && (
                <>
                    <div className="sample-info" style={{ marginBottom: '30px' }}>
                        <h3>PATIENT INFORMATION</h3>
                        <p style={{ fontSize: '16px', color: '#0f0' }}>{fullCheckData.patient_metadata}</p>
                    </div>

                    {/* Top 5 Most Likely */}
                    <div className="result-card" style={{ marginBottom: '30px', background: 'linear-gradient(135deg, #1a0a0a 0%, #0a1a1a 100%)', borderColor: '#ff0000' }}>
                        <h2 style={{ color: '#ff0000', marginBottom: '20px' }}>🔴 TOP 5 MOST LIKELY DISORDERS</h2>
                        {fullCheckData.top_disorders.map((disorder, idx) => (
                            <div key={disorder.disorder} style={{
                                padding: '15px',
                                marginBottom: '15px',
                                background: idx === 0 ? 'rgba(255, 0, 0, 0.2)' : 'rgba(0, 255, 136, 0.1)',
                                border: `2px solid ${idx === 0 ? '#ff0000' : '#00ff88'}`,
                                borderRadius: '8px'
                            }}>
                                <div style={{ fontSize: '18px', fontWeight: 'bold', color: idx === 0 ? '#ff0000' : '#00ff88', marginBottom: '10px' }}>
                                    #{idx + 1}: {disorder.disorder.replace(/_/g, ' ')}
                                </div>
                                <div className="metric">
                                    <span className="metric-label">PREDICTION:</span>
                                    <span className="metric-value" style={{ color: disorder.prediction === 1 ? '#ff0000' : '#0f0' }}>
                                        {disorder.prediction === 1 ? 'PRESENT' : 'ABSENT'}
                                    </span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">PROBABILITY:</span>
                                    <span className="metric-value">{(disorder.probability * 100).toFixed(2)}%</span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">CONFIDENCE:</span>
                                    <span className="metric-value">{disorder.confidence}</span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">KNN:</span>
                                    <span className="metric-value">{(disorder.individual_predictions.knn * 100).toFixed(1)}%</span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">RANDOM FOREST:</span>
                                    <span className="metric-value">{(disorder.individual_predictions.random_forest * 100).toFixed(1)}%</span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">EXTRA TREES:</span>
                                    <span className="metric-value">{(disorder.individual_predictions.extra_trees * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* AI Explanation */}
                    <AIExplanationCard explanation={fullCheckData.ai_explanation} loading={loading} />

                    {/* All Results Table */}
                    <div className="result-card" style={{ marginTop: '30px' }}>
                        <h2 style={{ marginBottom: '20px' }}>📋 COMPLETE SCREENING RESULTS ({fullCheckData.total_checked} DISORDERS)</h2>
                        <div style={{ overflowX: 'auto' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                <thead>
                                    <tr style={{ borderBottom: '2px solid #00ff88' }}>
                                        <th style={{ padding: '12px', textAlign: 'left', color: '#00ff88' }}>DISORDER</th>
                                        <th style={{ padding: '12px', textAlign: 'center', color: '#00ff88' }}>PREDICTION</th>
                                        <th style={{ padding: '12px', textAlign: 'center', color: '#00ff88' }}>PROBABILITY</th>
                                        <th style={{ padding: '12px', textAlign: 'center', color: '#00ff88' }}>CONFIDENCE</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {fullCheckData.all_predictions.map((disorder, idx) => (
                                        <tr key={disorder.disorder} style={{
                                            borderBottom: '1px solid #333',
                                            background: idx < 5 ? 'rgba(0, 255, 136, 0.05)' : 'transparent'
                                        }}>
                                            <td style={{ padding: '12px' }}>{disorder.disorder.replace(/_/g, ' ')}</td>
                                            <td style={{ padding: '12px', textAlign: 'center', color: disorder.prediction === 1 ? '#ff0000' : '#666' }}>
                                                {disorder.prediction === 1 ? 'PRESENT' : 'ABSENT'}
                                            </td>
                                            <td style={{ padding: '12px', textAlign: 'center', fontWeight: 'bold' }}>
                                                {(disorder.probability * 100).toFixed(2)}%
                                            </td>
                                            <td style={{ padding: '12px', textAlign: 'center' }}>{disorder.confidence}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}

// Mount the app
ReactDOM.render(<App />, document.getElementById('root'));
