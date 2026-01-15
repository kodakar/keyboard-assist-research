platex master_yokou.tex
pbibtex master_yokou.aux
platex master_yokou.tex
platex master_yokou.tex
dvipdfmx master_yokou.dvi
open master_yokou.pdf
rm *.blg
rm *.log
rm *.dvi
rm *.bbl
rm *.aux

