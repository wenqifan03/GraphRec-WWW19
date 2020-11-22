Data Collection Duration: 2013-11 --- 2013-12
Author: Guibing Guo

========================================= Collected Data Formats ==================================
Dataset Name: CiaoDVDs

1. Ciao movie ratings format:  
    1) File: movie-ratings.txt (size: 72,665 --> 72.7K)
    2) Columns: userID, movieID, genreID, reviewID, movieRating, date

2. Ciao review ratings format: 
    1) File: review-ratings.txt (size: 1,625,480 --> 1.6M)
    2) Columns: userID, reviewID, reviewRating
    3) Note: There are users who do not provide movie ratings but provide review ratings.
    
3. Ciao user trusts fromat:
    1) File: trusts.txt (size: 40,133 --> 40K)
    2) Columns: trustorID, trusteeID, trustValue
    3) Note: There are users who may not provide neither movie rating nor review ratings. 

4. We will use/release the dataset with anonymous ids. 

5. Reference: 

    @INPROCEEDINGS{guo2014etaf,
      author = {Guo, G. and Zhang, J. and Thalmann, D. and Yorke-Smith, N.},
      title = {ETAF: An Extended Trust Antecedents Framework for Trust Prediction},
      booktitle = {Proceedings of the 2014 International Conference on Advances in Social Networks Analysis and Mining (ASONAM)},
      year = {2014}
    } 

========================================= Data Sources ================================================
Ciao category: DVDs (http://dvd.ciao.co.uk/) including 17 genres:
    1) Action & Adventure: http://dvd.ciao.co.uk/Action_Adventure_5291510_2
    2) Comedy: http://dvd.ciao.co.uk/Comedy_5291530_2
    3) Family: http://dvd.ciao.co.uk/Family_5291512_2
    4) Drama: http://dvd.ciao.co.uk/Drama_5291532_2
    5) Horror: http://dvd.ciao.co.uk/Horror_5291535_2
    6) Science Fiction & Fantasy: http://dvd.ciao.co.uk/Science_Fiction_Fantasy_5291533_2
    7) Thriller & Mystery: http://dvd.ciao.co.uk/Thriller_Mystery_5291538_2
    8) Martial Arts: http://dvd.ciao.co.uk/Martial_Arts_5291536_2
    9) Musicals & Music Films: http://dvd.ciao.co.uk/Musicals_Music_Films_5291537_2
    10) War: http://dvd.ciao.co.uk/War_5291540_2
    11) Westerns: http://dvd.ciao.co.uk/Westerns_5291541_2
    12) Documentaries & Biographies: http://dvd.ciao.co.uk/Documentaries_Biographies_5291531_2
    13) Special Interest: http://dvd.ciao.co.uk/Special_Interest_5291534_2
    14) Sports: http://dvd.ciao.co.uk/Sports_5291539_2
    15) World Cinema: http://dvd.ciao.co.uk/World_Cinema_5302314_2
    16) TV Series: http://dvd.ciao.co.uk/TV_Series_5302317_2
    17) Anime: http://dvd.ciao.co.uk/Anime_5303111_2

========================================= Product Information ========================================
1. DVD description
2. Product details 
    1) Actors
    2) Directors
    3) Genre
    4) Classification
    5) Product Year
    6) Colour
    7) Running Time
    8) Franchise Name
    9) Video Category
    10) Country of Origin
    11) Plot
    12) Main Language
3. Release Details
    1) DVD Region
    2) Studios
    3) Release date
4. Ciao
    1) Listed on Ciao since (creation time)
    
Note: We throw aways products without any user reviews. 
    
========================================= Review Information ========================================
Writing on Ciao pays:
    1) Your experience helps others to make the right choices
    2) Knowledge is cash: earn	 up to ¡ê3 for each review
    3) Receive feedback and recognition for your contributions

For each specific review: 

1. Overall rating
2. Detailed rating: 
    1) Story
    2) Characters/performances
    3) Special Effects
    4) How does it compare to similar audio books/films?
    5) Did you enjoy it?
3. Features
    1) Advantages
    2) Disadvantages
4. Reviwe Text Content
5. Review Author (user)
6. Review Helpfulness given by other users (see below)

---------------------------------------- Review Helpfulness Ratings --------------------------------
Rating - Meaning:
    0. Off topic
    1. Not Helpful
    2. Somewhat Helpful (less helpful)
    3. Helpful
    4. Very Helpful
    5. Exceptional

Helpfulness Ratings
    1) Rater (user)
    2) Helpfulness Rating
    3) Timestamp?

========================================= User Information ========================================
1. User Profile:
    1) Name
    2) Member since
    3) Ciao id

2. Trust Circle: Buddies
    1) Members who trust the user (LinksFrom.txt)
    2) Members who the user trusts (LinksTo.txt)