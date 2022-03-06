import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from nltk.stem import SnowballStemmer
import string

STEMMER = SnowballStemmer("english")

def transform_name(product_name):
    product_name = product_name.lower()
    product_name = ''.join([' ' if word in string.punctuation else word for word in product_name]) # punctuation to spaces
    product_name = re.sub(r"\d+", "", product_name) # remove numbers
    product_name = ' '.join([STEMMER.stem(word) for word in product_name.split()]) # remove multiple spaces and stem
    return product_name

def categories_with_minimum_products(directory, min_products_allowed=50, category_depth=2):
    products_cats = dict()
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                    if len(child.find('categoryPath')) >= category_depth + 1:
                        cat = child.find('categoryPath')[category_depth][0].text
                    else:
                        cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                    if cat in products_cats:
                        products_cats[cat] += 1
                    else:
                        products_cats[cat] = 1
    print(f"Products categories distribution:\n{products_cats}")
    allowed_cats = [k for k, v in products_cats.items() if v>=min_products_allowed]
    return allowed_cats

    

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")
general.add_argument("--category_depth", default=2, type=int, help="The maximum depth when choosing a category (default is 2).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
min_products = args.min_products
category_depth = args.category_depth
allowed_cats = categories_with_minimum_products(directory, min_products_allowed=min_products, category_depth=category_depth)
sample_rate = args.sample_rate


print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print("Processing %s" % filename)
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                    if len(child.find('categoryPath')) >= category_depth + 1:
                        cat = child.find('categoryPath')[category_depth][0].text
                    else:
                        cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                    if cat not in allowed_cats:
                        continue
                    else:
                        name = child.find('name').text.replace('\n', ' ')
                        output.write("__label__%s %s\n" % (cat, transform_name(name)))

