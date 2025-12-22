# awrextractor

awrextractor.py - AWR-Miner text file extractor to pandas DataFrames and CSVs

Copyright (C) 2025 Irvansyah(Cunkrink)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

## Authors

- [@cunkrinks](https://www.github.com/cunkrinks)


## 🚀 About Me
I'm not a developer, i just love to code


## License

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)


    
## Usage/Examples

Extract every ~~BEGIN-...~~ / ~~END-...~~ block from an AWR-like text file
and convert blocks that contain dash-delimited column lines into
pandas.DataFrame objects. The first dash row is used to compute column
boundaries and then removed from data.

Usage examples:

    py .\awrtest2.py filename.out --section SGA --outdir out_all --csv
    py .\awrtest2.py filename.out --csv-all --outdir out_all


    py .\awrextractor.py --help
    usage: awrextractor.py [-h] [--section SECTION] [--outdir OUTDIR] [--csv] [--csv-all] [--excel] [--excel-filename EXCEL_FILENAME]   [--verbose] INPUT
    
    positional arguments:
      INPUT                 Input file path or name (required). If a path is provided it will be used; otherwise the filename is resolved   relative to the current working
                            directory.
    
    options:
      -h, --help            show this help message and exit
      --section, -s SECTION
                            Only extract this named section
      --outdir, -o OUTDIR
      --csv                 Write CSV for extracted sections
      --csv-all             Write CSV for all sections
      --excel               Write all sections to a single Excel file
      --excel-filename EXCEL_FILENAME
                            Excel output filename (default: awr_extracted_sections.xlsx)
      --verbose, -v         More detialed output


## TODO list

todo:
- populate data for analysis databese performance tuning:
   * Cpu usage
   * average active sessions
   * db time
    * top sql by cpu
   * Wait events
   * I/O stats
   * logswitches
   * Memory usage
   * capacity planning