const { useState, useEffect } = React;

const API_BASE_URL = 'http://localhost:8000';

// Main App Component
function App() {
    const [disorders, setDisorders] = useState([]);
    const [selectedDisorder, setSelectedDisorder] = useState('');
    const [sampleData, setSampleData] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [shapData, setShapData] = useState(null);
    const [limeData, setLimeData] = useState(null);
    const [featureSpace, setFeatureSpace] = useState(null);
    const [decisionPath, setDecisionPath] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Fetch disorders on mount
    useEffect(() => {
        fetchDisorders();
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

    const fetchSample = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/sample`);
            const data = await response.json();
            setSampleData(data);
        } catch (err) {
            setError('Failed to fetch sample data');
        }
    };

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: selectedDisorder })
            });
            const data = await response.json();
            setPrediction(data);
        } catch (err) {
            setError('Prediction failed: ' + err.message);
        }
        setLoading(false);
    };

    const handleSHAP = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/explain/shap`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: selectedDisorder })
            });
            const data = await response.json();
            setShapData(data);
        } catch (err) {
            setError('SHAP explanation failed: ' + err.message);
        }
        setLoading(false);
    };

    const handleLIME = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/explain/lime`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: selectedDisorder })
            });
            const data = await response.json();
            setLimeData(data);
        } catch (err) {
            setError('LIME explanation failed: ' + err.message);
        }
        setLoading(false);
    };

    const handleFeatureSpace = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/visualize/feature_space`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: selectedDisorder })
            });
            const data = await response.json();
            setFeatureSpace(data);
        } catch (err) {
            setError('Feature space visualization failed: ' + err.message);
        }
        setLoading(false);
    };

    const handleDecisionPath = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/visualize/decision_path`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disorder: selectedDisorder })
            });
            const data = await response.json();
            setDecisionPath(data);
        } catch (err) {
            setError('Decision path visualization failed: ' + err.message);
        }
        setLoading(false);
    };

    return (
        <div className="app">
            <header className="header">
                <h1>🧠 EEG Model Explainability Dashboard</h1>
                <p>Understanding AI Predictions for Psychiatric Disorder Classification</p>
            </header>

            <div className="controls">
                <div className="control-group">
                    <label>Select Disorder:</label>
                    <select
                        value={selectedDisorder}
                        onChange={(e) => setSelectedDisorder(e.target.value)}
                    >
                        {disorders.map(d => (
                            <option key={d.safe_name} value={d.safe_name}>
                                {d.name} (F1: {(d.f1_score * 100).toFixed(1)}%)
                            </option>
                        ))}
                    </select>
                </div>

                <div className="button-group">
                    <button className="btn btn-primary" onClick={handlePredict} disabled={loading}>
                        🎯 Predict
                    </button>
                    <button className="btn btn-secondary" onClick={handleSHAP} disabled={loading}>
                        📊 SHAP Explanation
                    </button>
                    <button className="btn btn-success" onClick={handleLIME} disabled={loading}>
                        🔍 LIME Explanation
                    </button>
                    <button className="btn btn-info" onClick={handleFeatureSpace} disabled={loading}>
                        🗺️ Feature Space
                    </button>
                    <button className="btn btn-info" onClick={handleDecisionPath} disabled={loading}>
                        🌳 Decision Path
                    </button>
                </div>

                {sampleData && <SampleInfo data={sampleData} />}
            </div>

            {loading && <div className="loading">⏳ Processing...</div>}
            {error && <div className="error">❌ {error}</div>}

            <div className="results">
                {prediction && <PredictionCard data={prediction} />}
                {shapData && <SHAPCard data={shapData} />}
                {limeData && <LIMECard data={limeData} />}
                {featureSpace && <FeatureSpaceCard data={featureSpace} />}
                {decisionPath && <DecisionPathCard data={decisionPath} />}
            </div>
        </div>
    );
}

// Sample Info Component
function SampleInfo({ data }) {
    return (
        <div className="sample-info">
            <h3>📋 Sample Patient Information</h3>
            <div className="info-grid">
                {data.metadata.no && (
                    <div className="info-item">
                        <div className="info-label">Patient ID</div>
                        <div className="info-value">#{data.metadata.no}</div>
                    </div>
                )}
                {data.metadata.sex && (
                    <div className="info-item">
                        <div className="info-label">Sex</div>
                        <div className="info-value">{data.metadata.sex}</div>
                    </div>
                )}
                {data.metadata.age && (
                    <div className="info-item">
                        <div className="info-label">Age</div>
                        <div className="info-value">{data.metadata.age}</div>
                    </div>
                )}
                <div className="info-item">
                    <div className="info-label">Features</div>
                    <div className="info-value">{data.num_features}</div>
                </div>
                {data.metadata.main_disorder && (
                    <div className="info-item">
                        <div className="info-label">Actual Disorder</div>
                        <div className="info-value">{data.metadata.main_disorder}</div>
                    </div>
                )}
            </div>
        </div>
    );
}

// Prediction Card Component
function PredictionCard({ data }) {
    const pred = data.prediction || data;
    const predictionLabel = pred.prediction === 1 ? 'POSITIVE' : 'NEGATIVE';
    const badgeClass = pred.prediction === 1 ? 'badge-positive' : 'badge-negative';

    return (
        <div className="card">
            <h2>🎯 Prediction Results</h2>

            <div className={`prediction-badge ${badgeClass}`}>
                {predictionLabel}
            </div>

            <div className="metric">
                <span className="metric-label">Probability:</span>
                <span className="metric-value">{(pred.probability * 100).toFixed(2)}%</span>
            </div>

            <div className="metric">
                <span className="metric-label">Threshold:</span>
                <span className="metric-value">{(pred.threshold * 100).toFixed(2)}%</span>
            </div>

            <div className="metric">
                <span className="metric-label">Confidence:</span>
                <span className={`metric-value confidence-${pred.confidence ? pred.confidence.toLowerCase() : 'medium'}`}>
                    {pred.confidence || 'N/A'}
                </span>
            </div>

            {pred.individual_predictions && (
                <>
                    <h3>Individual Model Predictions</h3>
                    <div className="metric">
                        <span className="metric-label">KNN:</span>
                        <span className="metric-value">{(pred.individual_predictions.knn * 100).toFixed(2)}%</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Random Forest:</span>
                        <span className="metric-value">{(pred.individual_predictions.random_forest * 100).toFixed(2)}%</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Extra Trees:</span>
                        <span className="metric-value">{(pred.individual_predictions.extra_trees * 100).toFixed(2)}%</span>
                    </div>
                </>
            )}

            {pred.performance_metrics && (
                <>
                    <h3>Model Performance</h3>
                    <div className="metric">
                        <span className="metric-label">F1-Score:</span>
                        <span className="metric-value">{(pred.performance_metrics.f1_score * 100).toFixed(2)}%</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Recall:</span>
                        <span className="metric-value">{(pred.performance_metrics.recall * 100).toFixed(2)}%</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Precision:</span>
                        <span className="metric-value">{(pred.performance_metrics.precision * 100).toFixed(2)}%</span>
                    </div>
                </>
            )}
        </div>
    );
}

// SHAP Card Component
function SHAPCard({ data }) {
    const shap = data.shap_values || data;

    if (shap.error) {
        return (
            <div className="card">
                <h2>📊 SHAP Explanation</h2>
                <div className="error">{shap.error}</div>
            </div>
        );
    }

    useEffect(() => {
        if (shap.top_features) {
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
                title: 'SHAP Feature Importance',
                xaxis: { title: 'SHAP Value' },
                yaxis: { title: '' },
                height: 500,
                margin: { l: 150, r: 50, t: 50, b: 50 },
                paper_bgcolor: '#000',
                plot_bgcolor: '#111',
                font: { color: '#fff' }
            };

            Plotly.newPlot('shap-chart', [trace], layout);
        }
    }, [shap]);

    return (
        <div className="card">
            <h2>📊 SHAP Explanation</h2>
            {shap.method && <p><strong>Method:</strong> {shap.method}</p>}

            {shap.base_value !== undefined && (
                <div className="metric">
                    <span className="metric-label">Base Value:</span>
                    <span className="metric-value">{shap.base_value.toFixed(4)}</span>
                </div>
            )}

            {shap.prediction !== undefined && (
                <div className="metric">
                    <span className="metric-label">Prediction:</span>
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
                    <h3>Top Features</h3>
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

// LIME Card Component
function LIMECard({ data }) {
    const lime = data.lime_values || data;

    if (lime.error) {
        return (
            <div className="card">
                <h2>🔍 LIME Explanation</h2>
                <div className="error">{lime.error}</div>
            </div>
        );
    }

    useEffect(() => {
        if (lime.top_features) {
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
                title: 'LIME Feature Importance',
                xaxis: { title: 'LIME Weight' },
                yaxis: { title: '' },
                height: 500,
                margin: { l: 150, r: 50, t: 50, b: 50 },
                paper_bgcolor: '#000',
                plot_bgcolor: '#111',
                font: { color: '#fff' }
            };

            Plotly.newPlot('lime-chart', [trace], layout);
        }
    }, [lime]);

    return (
        <div className="card">
            <h2>🔍 LIME Explanation</h2>
            {lime.method && <p><strong>Method:</strong> {lime.method}</p>}

            {lime.prediction_probability !== undefined && (
                <div className="metric">
                    <span className="metric-label">Prediction Probability:</span>
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
                    <h3>Top Features</h3>
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

// Feature Space Card Component
function FeatureSpaceCard({ data }) {
    const feat = data.visualization || data;

    if (feat.error) {
        return (
            <div className="card">
                <h2>🗺️ Feature Space Visualization</h2>
                <div className="error">{feat.error}</div>
            </div>
        );
    }

    useEffect(() => {
        if (feat.coordinates) {
            const trace = {
                type: 'scatter',
                mode: 'markers',
                x: [feat.coordinates.pc1],
                y: [feat.coordinates.pc2],
                marker: {
                    size: 20,
                    color: '#00ff00',
                    symbol: 'star',
                    line: {
                        color: '#fff',
                        width: 2
                    }
                },
                text: ['Sample Patient'],
                hoverinfo: 'text'
            };

            const layout = {
                title: 'PCA Feature Space',
                xaxis: {
                    title: `PC1 (${feat.explained_variance ? (feat.explained_variance.pc1 * 100).toFixed(1) : '?'}% variance)`,
                    zeroline: true,
                    gridcolor: '#333'
                },
                yaxis: {
                    title: `PC2 (${feat.explained_variance ? (feat.explained_variance.pc2 * 100).toFixed(1) : '?'}% variance)`,
                    zeroline: true,
                    gridcolor: '#333'
                },
                height: 500,
                paper_bgcolor: '#000',
                plot_bgcolor: '#111',
                font: { color: '#fff' }
            };

            Plotly.newPlot('pca-chart', [trace], layout);
        }
    }, [feat]);

    return (
        <div className="card">
            <h2>🗺️ Feature Space Visualization</h2>
            {feat.method && <p><strong>Method:</strong> {feat.method}</p>}

            {feat.coordinates && (
                <>
                    <div className="metric">
                        <span className="metric-label">PC1 Coordinate:</span>
                        <span className="metric-value">{feat.coordinates.pc1.toFixed(4)}</span>
                    </div>

                    <div className="metric">
                        <span className="metric-label">PC2 Coordinate:</span>
                        <span className="metric-value">{feat.coordinates.pc2.toFixed(4)}</span>
                    </div>
                </>
            )}

            {feat.explained_variance && (
                <div className="metric">
                    <span className="metric-label">Total Variance Explained:</span>
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

// Decision Path Card Component
function DecisionPathCard({ data }) {
    const dec = data.decision_path || data;

    if (dec.error) {
        return (
            <div className="card">
                <h2>🌳 Decision Path</h2>
                <div className="error">{dec.error}</div>
            </div>
        );
    }

    useEffect(() => {
        if (dec.random_forest && dec.random_forest.top_features) {
            const features = dec.random_forest.top_features.slice(0, 15);
            const trace = {
                type: 'bar',
                x: features.map(f => f.importance),
                y: features.map(f => `Feature ${f.index}`),
                orientation: 'h',
                marker: {
                    color: '#ffff00',
                }
            };

            const layout = {
                title: 'Random Forest Feature Importance',
                xaxis: { title: 'Importance' },
                yaxis: { title: '' },
                height: 500,
                margin: { l: 150, r: 50, t: 50, b: 50 },
                paper_bgcolor: '#000',
                plot_bgcolor: '#111',
                font: { color: '#fff' }
            };

            Plotly.newPlot('decision-chart', [trace], layout);
        }
    }, [dec]);

    return (
        <div className="card">
            <h2>🌳 Decision Path Analysis</h2>
            {dec.method && <p><strong>Method:</strong> {dec.method}</p>}

            {dec.interpretation && (
                <div className="interpretation">
                    <p>{dec.interpretation}</p>
                </div>
            )}

            {dec.random_forest && (
                <>
                    <h3>Random Forest</h3>
                    <div className="metric">
                        <span className="metric-label">Number of Trees:</span>
                        <span className="metric-value">{dec.random_forest.num_trees}</span>
                    </div>

                    <div className="metric">
                        <span className="metric-label">Example Path Length:</span>
                        <span className="metric-value">{dec.random_forest.path_length} nodes</span>
                    </div>
                </>
            )}

            <div id="decision-chart" className="chart-container"></div>

            {dec.knn_neighbors && (
                <>
                    <h3>KNN Neighbors Analysis</h3>
                    <div className="metric">
                        <span className="metric-label">K Value:</span>
                        <span className="metric-value">{dec.knn_neighbors.k}</span>
                    </div>

                    <div className="interpretation">
                        <p>{dec.knn_neighbors.interpretation}</p>
                    </div>
                </>
            )}
        </div>
    );
}

// Render App
ReactDOM.render(<App />, document.getElementById('root'));
