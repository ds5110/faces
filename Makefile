.PHONY: data

#-- Download the data
# mkdir -p fails quietly if directory already exists
# curl -L follows indirects
# curl -O preserves filename of source
data:
	mkdir -p data
	cd data &&\
	curl -LO https://coe.northeastern.edu/Research/AClab/InfAnFace/labels.csv &&\
	curl -LO https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/raw/master/data/joint/300w_infanface_train.csv &&\
	curl -LO https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/raw/master/data/300w/300w_valid.csv

clean:
	rm data/labels.csv &&\
	rm data/300w_infanface_train.csv &&\
	rm data/300w_valid.csv