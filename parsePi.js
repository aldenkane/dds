//Import parse module
const Parse = require('parse/node');

//Initialize Parse server
Parse.serverURL = 'https://optoswim.back4app.io'; // This is your Server URL
Parse.initialize(
  '03Pq0kbLRvci8D3OV92OFIIbNidw3kZrGma2sruS', // This is your Application ID
  'SMnIF1sMs1zdczYwWU1SikdLtfIu4IzcWYhBhEMf', // This is your Javascript key
  'HSNUMKsUeTtNjTxOIPB2ct3FIiD6NMJp7yc5w9WW' // This is your Master key (never use it in the frontend)
);

//Create object in Image class linked to specific user

function sendImage(swimmerDetected, numberSwimmers, drowningDetected){
  const Images = Parse.Object.extend('Images');
  const image = new Images();

  image.set('image', new Parse.File("lastFrame.jpg", { base64: btoa("My file content") }));
  image.set('swimmerDetected', swimmerDetected);
  image.set('numberSwimmers', numberSwimmers);
  image.set('Emergency', drowningDetected);
  iamge.set('user', Parse.User.current());

  myNewObject.save().then(
    console.log('The image has been sent to the Parse server, let's fucking go!)
  );
};
