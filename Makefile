.PHONY: data

#-- Download the data
# mkdir -p fails quietly if directory already exists
# curl -L follows indirects
# curl -O preserves filename of source
data:
	mkdir -p data
	cd data &&\
	curl -LO https://coe.northeastern.edu/Research/AClab/InfAnFace/labels.csv

clean:
	rm data/labels.csv