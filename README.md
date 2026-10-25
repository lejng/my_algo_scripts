# Useful commands

### 1. Determinate is it virt env or not
```
where pip
```

### 2. Activate virtual env or create virtual env
activate
``` 
# windows
.venv\Scripts\activate

# macos
source .venv/bin/activate
```
create:
```
python -m venv .venv
```

### 3. Install/remove dependencies
```
pip install -r requirements.txt

pip uninstall -r requirements.txt
```

### 4. Deactivate virtual env
```
deactivate
```

### 5. How to run console app
```
python -m src.main.volatility_squeeze_screener
python -m src.main.correlation_screener
python -m src.main.сross_sectional_momentum
python -m src.main.ratio.ratio_screener
python -m src.main.ratio.relative_strength

```
