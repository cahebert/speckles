#!/bin/

for i in 20190517 20190521 20190620 20190715 20190719 20191009 20190518 20190522 20190621 20190716 20190720 20190914 20191010 20190519 20190523 20190622 20190717 20190721 20190916 20190520 20190614 20190714 20190718 20190928 20191009 20191010
do
echo "analysing obs date "$i
python data_selection.py --stars='bright' --obsdate=$i
done
