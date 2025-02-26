document.addEventListener('DOMContentLoaded', function () {
    const textInput = document.getElementById('text-input');
    const numWords = document.getElementById('num-words');
    const numPredictions = document.getElementById('num-predictions');
    const predictBtn = document.getElementById('predict-btn');
    const resultsContainer = document.getElementById('results-container');
    const loadingMessage = document.getElementById('loading-message');
    const spinner = document.querySelector('.spinner-border');
    const modeToggle = document.getElementById('mode-toggle');

    // Dark mode toggle
    modeToggle.addEventListener('click', function () {
        document.body.classList.toggle('dark-mode');
        const icon = modeToggle.querySelector('i');
        if (icon.classList.contains('fa-moon')) {
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        } else {
            icon.classList.remove('fa-sun');
            icon.classList.add('fa-moon');
        }
    });

    // Handle prediction button click
    predictBtn.addEventListener('click', async function () {
        // Check if input is empty
        if (textInput.value.trim() === '') {
            resultsContainer.innerHTML = `
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Please enter some text first.
                        </div>
                    `;
            return;
        }

        // Show loading state
        predictBtn.disabled = true;
        spinner.style.display = 'inline-block';
        loadingMessage.style.display = 'flex';
        resultsContainer.style.display = 'none';

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: textInput.value.trim(),
                    num_words: parseInt(numWords.value),
                    num_predictions: parseInt(numPredictions.value)
                    // Temperature removed as requested
                }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            resultsContainer.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle me-2"></i>
                            Error: ${error.message}
                        </div>
                    `;
            resultsContainer.style.display = 'block';
        } finally {
            // Hide loading state
            predictBtn.disabled = false;
            spinner.style.display = 'none';
            loadingMessage.style.display = 'none';
            resultsContainer.style.display = 'block';
        }
    });

    // Display prediction results
    function displayResults(data) {
        const originalText = data.original_text;
        const predictions = data.predictions;
        const processingTime = data.processing_time.toFixed(3);

        let html = `
                    <div class="result-section">
                        <div class="d-flex align-items-center mb-3">
                            <i class="fas fa-clock me-2 text-muted"></i>
                            <span>Generated ${predictions.length} predictions in ${processingTime} seconds</span>
                        </div>
                        <div class="mb-4">
                            <h6><i class="fas fa-quote-left me-2 text-muted"></i>Original text:</h6>
                            <p class="lead">"${originalText}"</p>
                        </div>
                    </div>

                    <div class="result-section">
                        <h6 class="mb-3"><i class="fas fa-star me-2 text-muted"></i>Suggested next words:</h6>
                        <div class="row">
                `;

        predictions.forEach((pred, index) => {
            const percentage = (pred.confidence * 100).toFixed(1);
            const barWidth = `${Math.max(pred.confidence * 100, 5)}%`;

            html += `
                        <div class="col-md-6 mb-3">
                            <div class="prediction-card" onclick="appendPrediction('${pred.text}')">
                                <div class="card-body">
                                    <h5 class="card-title">
                                        <span class="prediction-number">${index + 1}</span>
                                        "${pred.text}"
                                    </h5>
                                    <p class="card-text text-muted">
                                        <i class="fas fa-chart-bar me-1"></i>
                                        Confidence: ${percentage}%
                                    </p>
                                    <div class="confidence-bar" style="width: ${barWidth}"></div>
                                </div>
                            </div>
                        </div>
                    `;
        });

        html += `
                    </div>
                    </div>

                    <div class="result-section">
                        <h6><i class="fas fa-check-circle me-2 text-muted"></i>Complete sentence with top prediction:</h6>
                        <div class="completed-sentence">
                            <p class="lead mb-0">"${originalText} ${predictions[0].text}"</p>
                        </div>
                        <div class="text-end mt-3">
                            <button class="btn btn-sm btn-outline-primary" onclick="appendPrediction('${predictions[0].text}')">
                                <i class="fas fa-plus-circle me-1"></i>Use this prediction
                            </button>
                        </div>
                    </div>
                `;

        resultsContainer.innerHTML = html;
    }

    // Add function to global scope for click handlers
    window.appendPrediction = function (text) {
        textInput.value = textInput.value.trim() + ' ' + text;
        textInput.focus();

        // Smooth scroll to text input area
        document.querySelector('.app-card').scrollIntoView({
            behavior: 'smooth'
        });
    };
});