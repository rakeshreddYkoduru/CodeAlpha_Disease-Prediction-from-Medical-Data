const API_BASE = 'http://localhost:5000';
let currentMetadata = {};

// UI Elements
const diseaseSelect = document.getElementById('disease-select');
const dynamicForm = document.getElementById('dynamic-form');
const predictBtn = document.getElementById('predict-btn');
const resultsDisplay = document.getElementById('results-display');
const initialMessage = document.getElementById('initial-message');
const algoResults = document.getElementById('algo-results');

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
    'Pregnancies': 'Pregnancies',
    'Glucose': 'Glucose Level',
    'BloodPressure': 'Blood Pressure',
    'SkinThickness': 'Skin Thickness',
    'Insulin': 'Insulin Level',
    'BMI': 'BMI',
    'DiabetesPedigreeFunction': 'Diabetes Pedigree',
    'Age': 'Age'
};

async function init() {
    try {
        const res = await fetch(`${API_BASE}/metadata`);
        currentMetadata = await res.json();
        renderForm('heart_disease');
        renderPerformanceChart('heart_disease');
    } catch (err) {
        console.error('Failed to load metadata', err);
    }
}

function renderForm(disease) {
    const data = currentMetadata[disease] || {};
    const features = data.features || [];
    dynamicForm.innerHTML = '';
    
    features.forEach(feat => {
        const label = featureLabels[feat] || feat;
        const group = document.createElement('div');
        group.className = 'form-group animate-fade';
        group.innerHTML = `
            <label>${label}</label>
            <input type="number" step="any" name="${feat}" placeholder="Enter ${label}" required>
        `;
        dynamicForm.appendChild(group);
    });
}

function renderPerformanceChart(disease) {
    const data = currentMetadata[disease] || {};
    const metrics = data.metrics || {};
    const algos = Object.keys(metrics);
    const accuracies = Object.values(metrics);

    const trace = {
        x: algos,
        y: accuracies,
        type: 'bar',
        text: accuracies.map(a => a + '%'),
        textposition: 'auto',
        marker: {
            color: '#6366f1',
            opacity: 0.7,
            line: { color: '#818cf8', width: 1.5 }
        }
    };

    const layout = {
        title: 'Training Accuracy (%)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8' },
        margin: { t: 40, b: 40, l: 40, r: 20 },
        yaxis: { range: [0, 100], gridcolor: 'rgba(255,255,255,0.1)' },
        xaxis: { gridcolor: 'rgba(255,255,255,0.1)' }
    };

    Plotly.newPlot('performance-chart', [trace], layout);
}

diseaseSelect.addEventListener('change', (e) => {
    const disease = e.target.value;
    renderForm(disease);
    renderPerformanceChart(disease);
});

predictBtn.addEventListener('click', async () => {
    const disease = diseaseSelect.value;
    const inputs = dynamicForm.querySelectorAll('input');
    const features = Array.from(inputs).map(inp => parseFloat(inp.value));

    if (features.some(isNaN)) {
        alert('Please fill all fields with valid numbers.');
        return;
    }

    predictBtn.innerText = 'Calculating...';
    predictBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ disease, features })
        });
        
        const results = await res.json();
        displayResults(results);
    } catch (err) {
        alert('Error communicating with backend. Ensure Flask is running.');
    } finally {
        predictBtn.innerText = 'Analyze Risk';
        predictBtn.disabled = false;
    }
});

function displayResults(data) {
    initialMessage.style.display = 'none';
    resultsDisplay.style.display = 'block';
    algoResults.innerHTML = '';

    const algos = Object.keys(data);
    let avgProb = 0;

    algos.forEach(name => {
        const result = data[name];
        avgProb += result.probability;
        
        const card = document.createElement('div');
        card.className = 'algo-card animate-fade';
        const riskClass = result.prediction === 1 ? 'risk-high' : 'risk-low';
        
        card.innerHTML = `
            <div class="algo-name">${name}</div>
            <div class="algo-prob ${riskClass}">${result.probability}% Risk</div>
            <div class="algo-name">${result.prediction === 1 ? 'Positive' : 'Negative'}</div>
        `;
        algoResults.appendChild(card);
    });

    avgProb /= algos.length;
    
    // Update summary
    const status = document.getElementById('overall-status');
    const icon = document.getElementById('overall-icon');
    const desc = document.getElementById('overall-desc');
    
    if (avgProb > 50) {
        status.innerText = 'High Risk Detected';
        icon.innerText = '🚨';
        icon.style.color = '#ef4444';
        desc.innerText = `Average risk assessment across models is ${avgProb.toFixed(1)}%. Consult a specialist.`;
    } else {
        status.innerText = 'Low Risk Assessment';
        icon.innerText = '✅';
        icon.style.color = '#10b981';
        desc.innerText = `Average risk assessment across models is ${avgProb.toFixed(1)}%. Continue healthy habits.`;
    }

    // Chart
    const trace = {
        x: algos,
        y: algos.map(n => data[n].probability),
        type: 'bar',
        marker: { color: algos.map(n => data[n].probability > 50 ? '#ef4444' : '#10b981') }
    };
    
    const layout = {
        title: 'Risk Confidence by Algorithm',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8' },
        margin: { t: 40, b: 40, l: 30, r: 30 }
    };

    Plotly.newPlot('probability-chart', [trace], layout);
    window.scrollTo({ top: resultsDisplay.offsetTop, behavior: 'smooth' });
}

init();
