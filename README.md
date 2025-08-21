

  <h1>AutoML Project with FLAML</h1>

  <p>
    This project follows the Medium tutorial:
    <a href="https://medium.com/lumenore/hands-on-tutorial-on-automatic-machine-learning-with-flaml-2ac26d36b1b1">
      Hands-on Tutorial on Automatic Machine Learning with FLAML
    </a>
  </p>

  <hr />

  <h2 id="what-is-flaml">What is FLAML?</h2>
  <p>
    <strong>FLAML (Fast and Lightweight AutoML)</strong> is a compact, efficient Python library with a
    scikit-learn–style API. It automates both algorithm selection and hyperparameter tuning so you can
    focus on your data and objective rather than manual experimentation.
  </p>

  <h2 id="getting-started">Getting Started</h2>

  <h3>Installation</h3>
  <pre><code class="language-bash">pip install flaml</code></pre>

  <h3>Quick Start (Classification)</h3>
  <pre><code class="language-python">from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task="classification")  # use task="regression" for continuous targets

print("Best model:", automl.best_estimator)
print("Best params:", automl.best_config)
print("Best validation loss:", automl.best_loss)</code></pre>

  <h2 id="model-building">Model Building</h2>
  <p>
    This project uses <strong>FLAML</strong> to automatically train and tune multiple models on the dataset,
    requiring just a few lines of code:
  </p>

  <pre><code class="language-python">from flaml import AutoML

automl = AutoML()
automl.fit(
    X_train,
    y_train,
    task="classification",   # or "regression"
    time_budget=60,          # optional: seconds to search
    metric="accuracy",       # pick a metric that suits your task
)</code></pre>

  <p>During one run, FLAML explored and tuned the following algorithms:</p>
  <ul>
    <li><code>lgbm</code></li>
    <li><code>rf</code> (Random Forest)</li>
    <li><code>catboost</code></li>
    <li><code>xgboost</code></li>
    <li><code>extra_tree</code></li>
    <li><code>lrl1</code> (Logistic Regression with L1)</li>
  </ul>

  <div class="callout">
    It selected the best model for the dataset automatically, finishing the search in just over
    <strong>34 seconds</strong> (as shown in the summary output of that run).
  </div>

  <h2 id="evaluate">Evaluate the Best Model</h2>
  <pre><code class="language-python">from sklearn.metrics import accuracy_score

y_pred = automl.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")</code></pre>

  <h2 id="reproducibility">Reproducibility Tips</h2>
  <ul>
    <li>Set a search time limit with <code>time_budget</code> to control runtime.</li>
    <li>Choose an evaluation metric (e.g., <code>accuracy</code>, <code>f1</code>, <code>r2</code>) aligned with your task.</li>
    <li>Fix random seeds in your data splits for consistent comparisons.</li>
  </ul>

  <h2 id="references">References</h2>
  <ul>
    <li>FLAML documentation: <a href="https://microsoft.github.io/FLAML/" target="_blank" rel="noopener">https://microsoft.github.io/FLAML/</a></li>
    <li>Tutorial followed: <a href="https://medium.com/lumenore/hands-on-tutorial-on-automatic-machine-learning-with-flaml-2ac26d36b1b1" target="_blank" rel="noopener">Medium post</a></li>
  </ul>

</body>
</html>
# AutoML_Project
