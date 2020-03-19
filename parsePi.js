//Import parse and btoa modules
const Parse = require('parse/node');
const btoa = require('btoa')

//Initialize Parse server
Parse.serverURL = 'https://optoswim.back4app.io'; // This is your Server URL
Parse.initialize(
  '03Pq0kbLRvci8D3OV92OFIIbNidw3kZrGma2sruS', // Application ID
  'SMnIF1sMs1zdczYwWU1SikdLtfIu4IzcWYhBhEMf', // Javascript key
  'HSNUMKsUeTtNjTxOIPB2ct3FIiD6NMJp7yc5w9WW' // Master key (never to be used in production on clientside)
);

//Create object in Image class linked to specific user

function sendImage(fileName, swimmerDetected, numberSwimmers, drowningDetected){
  const Images = Parse.Object.extend('Images');
  const image = new Images();

  image.set('image', new Parse.File(fileName, { base64: btoa(fileName) }));
  image.set('swimmerDetected', swimmerDetected);
  image.set('numberSwimmers', numberSwimmers);
  image.set('Emergency', drowningDetected);
  iamge.set('user', Parse.User.current());

  myNewObject.save().then(
    console.log('The image has been sent to the Parse server, let's fucking go!)
  );
};
