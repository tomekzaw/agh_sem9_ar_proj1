# n-body problem

```
python3 gen.py
./run.sh
python3 test.py
```

## TODO

- don't calculate own interactions twice in zad2
- why `(p+1)//2`

## Lessons Learned

### Communication

- `send` + `recv` &rarr; `UnpickleError`
- `send` + `Recv` &rarr; non-compatible due to different protocol
- `Send` + `Recv` &rarr; blocked for large payloads
- `Isend` + `Recv` + single `np.array` &rarr; Recv overwrites Isend array
- `Isend` + `Recv` + double buffering &rarr; works like charm

### Flags

- `--use-hwthread-cpus`
- `--oversubscribe`
