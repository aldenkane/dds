const Parse = require('parse/node');
const btoa = require('btoa');
const fs = require('fs');

Parse.serverURL = 'https://optoswim.back4app.io'; // This is your Server URL
Parse.initialize(
  '03Pq0kbLRvci8D3OV92OFIIbNidw3kZrGma2sruS', // This is your Application ID
  'SMnIF1sMs1zdczYwWU1SikdLtfIu4IzcWYhBhEMf', // This is your Javascript key
  'HSNUMKsUeTtNjTxOIPB2ct3FIiD6NMJp7yc5w9WW' // This is your Master key (never use it in the frontend)
);

function send(swimDetected, numberSwimmers, drownDetected, serialNo) {
  let file = fs.readFileSync('../last_Image/last_Frame.jpg');
  const Images = Parse.Object.extend('Images');
  const image = new Images();

  image.set('swimDetected', swimDetected);
  image.set('numberSwimmers', numberSwimmers);
  image.set('drownDetected', drownDetected);
  image.set('serialNo', serialNo);
  image.set('image', new Parse.File('last_Frame.jpg', { base64: btoa(file) }));

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
// const options = {
//   persistent: true,
// }
fs.watchFile('../last_Image/event.json', (event, filename) => {
  if (filename) {
    let jsObj = JSON.parse(fs.readFileSync('../last_Image/event.json', 'utf8'));
    send(
      jsObj.swimDetected,
      parseInt(jsObj.numberSwimmers),
      jsObj.drownDetected,
      jsObj.serialNo
    );
  }
});
