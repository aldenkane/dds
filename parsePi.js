const Parse = require('parse/node');
const btoa = require('btoa');
const fs = require('fs');

Parse.serverURL = 'https://optoswim.back4app.io'; // This is your Server URL
Parse.initialize(
  '03Pq0kbLRvci8D3OV92OFIIbNidw3kZrGma2sruS', // This is your Application ID
  'SMnIF1sMs1zdczYwWU1SikdLtfIu4IzcWYhBhEMf', // This is your Javascript key
  'HSNUMKsUeTtNjTxOIPB2ct3FIiD6NMJp7yc5w9WW' // This is your Master key (never use it in the frontend)
);

function send(swimDetected, numberSwimmers, drownDetected) {
  const Images = Parse.Object.extend('Images');
  const image = new Images();

  image.set('swimDetected', swimDetected);
  image.set('numberSwimmers', numberSwimmers);
  image.set('drownDetected', drownDetected);
  image.set(
    'image',
    new Parse.File('last_Frame.jpg', { base64: btoa('last_Frame.jpg') })
  );

  image.save().then(
    (result) => {
      if (typeof document !== 'undefined')
        document.write(`Images created: ${JSON.stringify(result)}`);
      console.log('Images created', result);
    },
    (error) => {
      if (typeof document !== 'undefined')
        document.write(`Error while creating Images: ${JSON.stringify(error)}`);
      console.error('Error while creating Images: ', error);
    }
  );
}

fs.watch('./event.json', (event, filename) => {
  if (filename) {
    let jsObj = JSON.parse(fs.readFileSync('./event.json'));
    send(
      jsObj.swimDetected,
      parseInt(jsObj.numberSwimmers),
      jsObj.drownDetected
    );
  }
});
