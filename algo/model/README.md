## Preprocess

Run

```
python split_data.py
```

## Pretrain

Run

```
python pretrain.py
```

## Search triangles

Run

```
python preprocess.py
g++ -O2 find_triangles.cpp -o find_triangles
./find_triangles
python save triangles.py
```
