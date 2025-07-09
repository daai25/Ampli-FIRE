import csv

with open('cleaning_raw_meta.csv', newline='\n') as f1, open('clean_data_genre.csv', newline='\n') as f2:
    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)

    try:
        row1 = next(reader1)
        for row2 in reader2:
            # Advance file1 until its first field matches file2
            while row1[0] != row2[0]:
                row1 = next(reader1)  # move to next line in file1

            # If match found
            print(f"{row1[1]}, {row1[2]}, \"",end="")
            for i in range(1, len(row2)):
                print(f'{row2[i][8::]};',end=" ")
            print("\"")
            row1 = next(reader1)  # advance file1 to prepare for next iteration

    except Exception as ex:
        print(ex)
