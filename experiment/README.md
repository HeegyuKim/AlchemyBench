# Prediction and evaluation for the example models
```bash
model=o3-mini-high
python predict.py --model $model --batch_size 200 --split test_high_impact
python judge.py "data/test_high_impact/$model/prediction_0209-v2.jsonl" --batch_size 200

# RAG
top_k=5
python predict.py --use_rag --top_k $top_k --batch_size 200 --model o3-mini
python judge.py data/test_high_impact/o3-mini/rag_0211__k${top_k}.jsonl --batch_size 200
```

