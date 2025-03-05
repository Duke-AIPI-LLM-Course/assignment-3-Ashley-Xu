
### Create a virtual environment
Windows

```
python3 -m venv .venv
.venv/Scripts/activate
```

Linux/Mac
```
python3 -m venv .venv
source .venv/bin/activate
```

### Install requirements
```
python3 -m pip install -r requirements.txt
```


### Running the Script without Streamlit
```
python3 rag.py "YOUR QUESTION HERE"
```

### Running the Streamlit App locally
```
streamlit run app.py
```
### Running the Eval Script
```
python3 eval.py
```
