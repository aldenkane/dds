//Import parse and btoa modules
const Parse = require('parse/node');
const btoa = require('btoa');
const fs = require('fs');

//Initialize Parse server
Parse.serverURL = 'https://optoswim.back4app.io'; // This is your Server URL
Parse.initialize(
  '03Pq0kbLRvci8D3OV92OFIIbNidw3kZrGma2sruS', // Application ID
  'SMnIF1sMs1zdczYwWU1SikdLtfIu4IzcWYhBhEMf', // Javascript key
  'HSNUMKsUeTtNjTxOIPB2ct3FIiD6NMJp7yc5w9WW' // Master key (never to be used in production on clientside)
);

//Create object in Image class linked to specific user
const sendImage = (imgFilePath) => {
    try{
        fs.watch(imgFilePath, (event, filename) => {
            if (filename){
            console.log(`${filename} file changed`)
            //Load JSON file into javscript object
            //let obj = JSON.parse(fs.readFileSync(jsonFilePath))
            const Images = Parse.Object.extend('Images');
            const image = new Images();
            //const imgFilePath = obj.imageFilePath
            image.set('image', new Parse.File(imgFilePath, { base64: btoa(imgFilePath) }));
            // image.set('swimDetected', obj.swimmerDetected);
            //image.set('numberSwimmers', obj.numberSwimmers);
            //image.set('drownDetected', obj.drowningDetected);
            //image.set('user', Parse.User.current());
            image.save()
            }
        })
    } catch (ex) {
        console.log(ex);
    }
};
 
sendImage('./last_Image/last_Frame.jpg');
 

sendImage('./last_Image/last_Frame.jpg');
