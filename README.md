# BeachImageClassification
Creating a deep learning model that classifies beach into "clean" or "dirty" categories. Inspired by this study that created a "Clean-Coast Detector": https://koreascience.kr/article/JAKO202014862061074.page.

## The Model
The model is a Keras model that utilizes a pretrained MobileNetV2 (https://arxiv.org/abs/1801.04381). I used https://www.tensorflow.org/tutorials/images/transfer_learning as a guide for implementation. After finetuning, the model achieved a validation accuracy of 89.13%. On a test set of 20 images, it achieved an accuracy of 95%. The model is available via the saved_model.zip.

## Web Scraping
Images from dataset collected via Google Image search and JavaScript url scraping. Clean beach data collected through Google search of "beach", and dirty beach data collected through Google search of "dirty beach". Tutorial to url scrape: https://pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/. JavaScript methods working as of July 21, 2022.

To save images of a google search: 
- scroll-past/load-in all images in Google you'd like to save into the dataset
- Open Java Console (in Mac: cmd + opt + j)
- Paste the following code:
```
/**
 * simulate a right-click event so we can grab the image URL using the
 * context menu alleviating the need to navigate to another page
 *
 * attributed to @jmiserez: http://pyimg.co/9qe7y
 *
 * @param   {object}  element  DOM Element
 *
 * @return  {void}
 */
function simulateRightClick( element ) {
    var event1 = new MouseEvent( 'mousedown', {
        bubbles: true,
        cancelable: false,
        view: window,
        button: 2,
        buttons: 2,
        clientX: element.getBoundingClientRect().x,
        clientY: element.getBoundingClientRect().y
    } );
    element.dispatchEvent( event1 );
    var event2 = new MouseEvent( 'mouseup', {
        bubbles: true,
        cancelable: false,
        view: window,
        button: 2,
        buttons: 0,
        clientX: element.getBoundingClientRect().x,
        clientY: element.getBoundingClientRect().y
    } );
    element.dispatchEvent( event2 );
    var event3 = new MouseEvent( 'contextmenu', {
        bubbles: true,
        cancelable: false,
        view: window,
        button: 2,
        buttons: 0,
        clientX: element.getBoundingClientRect().x,
        clientY: element.getBoundingClientRect().y
    } );
    element.dispatchEvent( event3 );
}

/**
 * grabs a URL Parameter from a query string because Google Images
 * stores the full image URL in a query parameter
 *
 * @param   {string}  queryString  The Query String
 * @param   {string}  key          The key to grab a value for
 *
 * @return  {string}               value
 */
function getURLParam( queryString, key ) {
    var vars = queryString.replace( /^\?/, '' ).split( '&' );
    for ( let i = 0; i < vars.length; i++ ) {
        let pair = vars[ i ].split( '=' );
        if ( pair[0] == key ) {
            return pair[1];
        }
    }
    return false;
}

/**
 * Generate and automatically download a txt file from the URL contents
 *
 * @param   {string}  contents  The contents to download
 *
 * @return  {void}
 */
function createDownload( contents ) {
    var hiddenElement = document.createElement( 'a' );
    hiddenElement.href = 'data:attachment/text,' + encodeURI( contents );
    hiddenElement.target = '_blank';
    hiddenElement.download = 'urls.csv';
    hiddenElement.click();
}

/**
 * grab all URLs va a Promise that resolves once all URLs have been
 * acquired
 *
 * @return  {object}  Promise object
 */
function grabUrls() {
    var urls = [];
    return new Promise( function( resolve, reject ) {
        var count = document.querySelectorAll(
        	'.isv-r a:first-of-type' ).length,
            index = 0;
        Array.prototype.forEach.call( document.querySelectorAll(
        	'.isv-r a:first-of-type' ), function( element ) {
            // using the right click menu Google will generate the
            // full-size URL; won't work in Internet Explorer
            // (http://pyimg.co/byukr)
            simulateRightClick( element.querySelector( ':scope img' ) );
            // Wait for it to appear on the <a> element
            var interval = setInterval( function() {
                if ( element.href.trim() !== '' ) {
                    clearInterval( interval );
                    // extract the full-size version of the image
                    let googleUrl = element.href.replace( /.*(\?)/, '$1' ),
                        fullImageUrl = decodeURIComponent(
                        	getURLParam( googleUrl, 'imgurl' ) );
                    if ( fullImageUrl !== 'false' ) {
                        urls.push( fullImageUrl );
                    }
                    // sometimes the URL returns a "false" string and
                    // we still want to count those so our Promise
                    // resolves
                    index++;
                    if ( index == ( count - 1 ) ) {
                        resolve( urls );
                    }
                }
            }, 10 );
        } );
    } );
}

/**
 * Call the main function to grab the URLs and initiate the download
 */
grabUrls().then( function( urls ) {
    urls = urls.join( ', ' );
    createDownload( urls );
} );
```

## Validating Images
Once image URLs are collected and loaded as image files into a folder by class, it is VERY IMPORTANT to validate images. If no validation is done, Keras, FastAI, and every other deep learning framework gets very upsetti (spaghetti). Here's what I did (and this is in the BeachImageClassification notebook as well):
    - Note: this code is in Python and is based off the comments of this StackOverflow post -> https://stackoverflow.com/questions/65438156/tensorflow-keras-error-unknown-image-file-format-one-of-jpeg-png-gif-bmp-re

```
import os
import cv2
import imghdr   # lets you identify what is the image type contained in file, byte stream or path-like object

def check_images(s_dir, ext_list):
    bad_images=[]
    bad_ext=[]
    s_list= os.listdir(s_dir)
    for klass in s_list:
        klass_path=os.path.join (s_dir, klass)
        print ('processing class directory ', klass)
        if os.path.isdir(klass_path):
            file_list=os.listdir(klass_path)
            for f in file_list:               
                f_path=os.path.join (klass_path,f)
                # This is our 'bad' image check - gets image type
                tip = imghdr.what(f_path)
                if ext_list.count(tip) == 0:
                  bad_images.append(f_path)
                # Another check to see if the image can be opened/read in
                if os.path.isfile(f_path):
                    try:
                        img=cv2.imread(f_path)
                        shape=img.shape
                    except:
                        print('file ', f_path, ' is not a valid image file')
                        bad_images.append(f_path)
                else:
                    print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
        else:
            print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
    return bad_images, bad_ext

# Location of your data classes
source_dir =r'/content/data'
good_exts=['jpg', 'png', 'jpeg', 'gif', 'bmp' ] # list of acceptable extensions
bad_file_list, bad_ext_list=check_images(source_dir, good_exts)
if len(bad_file_list) !=0:
    print('improper image files are listed below')
    for i in range (len(bad_file_list)):
        print(bad_file_list[i])
        # Load image
        image = cv2.imread(bad_file_list[i])
        if (image is not None):
          # Save as proper .jpg image
          cv2.imwrite(bad_file_list[i], image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        elif (os.path.exists(bad_file_list[i])):
          # Delete image if the file exists but is a None image type -> aka can't be converted to a jpg
          os.remove(bad_file_list[i])
          print("deleted file: " + bad_file_list[i])
else:
    print(' no improper image files were found')
```
