# datamill

A command-line tool for transforming CSV data through composable pipelines.

## Overview

datamill lets you chain data transformation operations on CSV files. You can filter rows, rename columns, compute derived fields, and output the result to a new file or stdout. It is designed for local data wrangling tasks where setting up a full ETL stack would be overkill.

## Installation

Install with pip:

```
pip install datamill
```

Or from source:

```
git clone https://github.com/example/datamill
cd datamill
pip install -e .
```

## Usage

Basic usage:

```
datamill run --input data.csv --pipeline pipeline.yaml --output result.csv
```

You can also pipe from stdin:

```
cat data.csv | datamill run --pipeline pipeline.yaml
```

### Pipeline Files

Pipelines are defined in YAML. Each step specifies an operation and its parameters:

```yaml
steps:
  - op: filter
    column: status
    value: active
  - op: rename
    mappings:
      old_name: new_name
  - op: derive
    column: full_name
    expression: "..."
```

The `derive` step supports basic expressions. See the expressions documentation for the full syntax.

## Operations

datamill supports the following operations:

- `filter` — keep rows matching a condition
- `rename` — rename one or more columns
- `derive` — add a computed column
- `drop` — remove columns
- `sort` — sort rows by a column
- `head` — keep the first N rows

Each operation is documented in the operations reference.

## Configuration

datamill reads from `~/.datamill/config.yaml` if present. Available settings:

```yaml
default_delimiter: ","
null_value: ""
encoding: utf-8
```

You can override settings per-run with `--config path/to/config.yaml`.

## Contributing

Contributions are welcome. Open a pull request against the main branch. Please include tests for any new operations.

## License

MIT
