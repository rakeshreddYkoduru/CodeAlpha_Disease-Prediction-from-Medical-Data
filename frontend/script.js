const API_BASE = 'http://localhost:4000';
let currentMetadata = {};

// UI Elements
const diseaseSelect = document.getElementById('disease-select');
const dynamicForm = document.getElementById('dynamic-form');
const predictBtn = document.getElementById('predict-btn');
const resultsDisplay = document.getElementById('results-display');
const initialMessage = document.getElementById('initial-message');
const algoResults = document.getElementById('algo-results');
const metricsList = document.getElementById('model-metrics-list');
const systemStatus = document.getElementById('system-status');

const featureLabels = {
    'age': 'Age (Years)',
    'sex': 'Sex (1=M, 0=F)',
    'cp': 'Chest Pain Type (0-3)',
    'trestbps': 'Resting BP (mm Hg)',
    'chol': 'Cholesterol (mg/dl)',
    'fbs': 'Fasting Sugar > 120 (1=T, 0=F)',
    'restecg': 'Resting ECG (0-2)',
    'thalach': 'Max Heart Rate',
    'exang': 'Exercise Angina (1=Y, 0=N)',
    'oldpeak': 'ST Depression',
    'slope': 'ST Slope (0-2)',
    'ca': 'Major Vessels (0-4)',
    'thal': 'Thalassemia (0-3)',
    'Pregnancies': 'Number of Pregnancies',
    'Glucose': 'Glucose Concentration',
    'BloodPressure': 'Diastolic BP (mm Hg)',
    'SkinThickness': 'Triceps Fold (mm)',
    'Insulin': '2-Hour Insulin (mu U/ml)',
    'BMI': 'Body Mass Index',
    'DiabetesPedigreeFunction': 'Genetic Score',
    'Age': 'Age (Years)',
    'radius_mean': 'Radius (Mean)',
    'texture_mean': 'Texture (Mean)',
    'perimeter_mean': 'Perimeter (Mean)',
    'area_mean': 'Area (Mean)',
    'smoothness_mean': 'Smoothness (Mean)',
    'compactness_mean': 'Compactness (Mean)',
    'concavity_mean': 'Concavity (Mean)',
    'concave points_mean': 'Concave Points (Mean)',
    'symmetry_mean': 'Symmetry (Mean)',
    'fractal_dimension_mean': 'Fractal Dimension'
};

async function init() {
    try {
        const res = await fetch(`${API_BASE}/metadata`);
        if (!res.ok) throw new Error('Status not OK');
        currentMetadata = await res.json();
        
        // Initial setup
        renderForm(diseaseSelect.value);
        updateMetrics(diseaseSelect.value);
        
        systemStatus.querySelector('span').innerText = 'AI Core Ready';
        systemStatus.querySelector('.status-dot').style.background = '#10b981';
    } catch (err) {
        console.error('Failed to load metadata', err);
        systemStatus.querySelector('span').innerText = 'Backend Offline';
        systemStatus.querySelector('.status-dot').style.background = '#f43f5e';
    }
}

function renderForm(disease) {
    const data = currentMetadata[disease] || {};
    const features = data.features || [];
    
    // Smooth transition
    dynamicForm.style.opacity = '0';
    
    setTimeout(() => {
        dynamicForm.innerHTML = '';
        features.forEach((feat, index) => {
            const label = featureLabels[feat] || feat;
            const group = document.createElement('div');
            group.className = 'form-group fade-in';
            group.style.animationDelay = `${index * 0.05}s`;
            group.innerHTML = `
                <label class="input-label" style="font-size: 0.75rem">${label}</label>
                <input type="number" step="any" name="${feat}" placeholder="Value..." required>
            `;
            dynamicForm.appendChild(group);
        });
        dynamicForm.style.opacity = '1';
    }, 200);
}

function updateMetrics(disease) {
    const data = currentMetadata[disease] || {};
    const metrics = data.metrics || {};
    metricsList.innerHTML = '';

    Object.entries(metrics).forEach(([model, accuracy]) => {
        const item = document.createElement('div');
        item.className = 'fade-in';
        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; font-size: 0.875rem; margin-bottom: 0.25rem;">
                <span>${model}</span>
                <span style="color: var(--primary)">${accuracy}%</span>
            </div>
            <div class="metric-bar">
                <div class="metric-fill" style="width: ${accuracy}%"></div>
            </div>
        `;
        metricsList.appendChild(item);
    });
}

diseaseSelect.addEventListener('change', (e) => {
    const disease = e.target.value;
    renderForm(disease);
    updateMetrics(disease);
});

predictBtn.addEventListener('click', async () => {
    const disease = diseaseSelect.value;
    const inputs = dynamicForm.querySelectorAll('input');
    const features = Array.from(inputs).map(inp => parseFloat(inp.value));

    if (features.some(isNaN)) {
        showStatus('Please fill all fields', 'error');
        return;
    }

    predictBtn.innerHTML = '<span class="status-dot"></span> Analyzing...';
    predictBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ disease, features })
        });
        
        if (!res.ok) throw new Error('Prediction failed');
        
        const results = await res.json();
        displayResults(results);
    } catch (err) {
        showStatus('Connection Failed', 'error');
    } finally {
        predictBtn.innerText = 'Run Diagnostic Analysis';
        predictBtn.disabled = false;
    }
});

function displayResults(data) {
    initialMessage.style.display = 'none';
    resultsDisplay.style.display = 'block';
    resultsDisplay.classList.add('fade-in');
    
    algoResults.innerHTML = '';
    const algos = Object.keys(data);
    
    algos.forEach((name, index) => {
        const result = data[name];
        const status = result.prediction === 1 ? 'High Risk' : 'Healthy';
        const color = result.prediction === 1 ? 'var(--danger)' : 'var(--secondary)';
        
        const card = document.createElement('div');
        card.className = 'stat-card fade-in';
        card.style.animationDelay = `${index * 0.1}s`;
        card.innerHTML = `
            <span class="stat-label">${name}</span>
            <div class="stat-value" style="color: ${color}">${result.probability}%</div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem">
                ${status} Projection
            </div>
        `;
        algoResults.appendChild(card);
    });

    renderProbabilityChart(data);
    resultsDisplay.scrollIntoView({ behavior: 'smooth' });
}

function renderProbabilityChart(data) {
    const algos = Object.keys(data);
    const probabilities = algos.map(n => data[n].probability);
    
    const trace = {
        x: algos,
        y: probabilities,
        type: 'bar',
        marker: {
            color: probabilities.map(p => p > 50 ? '#f43f5e' : '#10b981'),
            line: { width: 0 }
        },
        bordercolor: 'transparent'
    };
    
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter', color: '#94a3b8' },
        margin: { t: 10, b: 40, l: 40, r: 10 },
        yaxis: { range: [0, 100], gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
        xaxis: { gridcolor: 'transparent' },
        showlegend: false
    };

    Plotly.newPlot('probability-chart', [trace], layout, { displayModeBar: false });
}

function showStatus(msg, type) {
    const originalText = predictBtn.innerText;
    predictBtn.innerText = msg;
    predictBtn.style.background = type === 'error' ? 'var(--danger)' : 'var(--secondary)';
    
    setTimeout(() => {
        predictBtn.innerText = 'Run Diagnostic Analysis';
        predictBtn.style.background = 'var(--primary)';
    }, 3000);
}

// Initial Call
init();
