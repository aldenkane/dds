const Parse = require('parse/node')
const btoa = require('btoa')
const fs = require('fs')

//Initializes Parse Object
Parse.serverURL = 'https://optoswim.back4app.io' // Server URL
Parse.initialize(
	'03Pq0kbLRvci8D3OV92OFIIbNidw3kZrGma2sruS', // Application ID
	'SMnIF1sMs1zdczYwWU1SikdLtfIu4IzcWYhBhEMf', // Javascript key
	'HSNUMKsUeTtNjTxOIPB2ct3FIiD6NMJp7yc5w9WW' // Master key (never use it in the frontend)
)

// Creates an Image object in Parse DB
function sendImage(swimDetected, numberSwimmers, drownDetected, serialNo) {
	let file = fs.readFileSync('../last_Image/last_Frame.jpg')
	const Images = Parse.Object.extend('Images')
	const image = new Images()

	image.set('swimDetected', swimDetected)
	image.set('numberSwimmers', numberSwimmers)
	image.set('drownDetected', drownDetected)
	image.set('serialNo', serialNo)
	image.set('image', new Parse.File('last_Frame.jpg', { base64: btoa(file) }))

	image.save().then(
		(result) => {
			if (typeof document !== 'undefined')
				document.write(`Images created: ${JSON.stringify(result)}`)
			console.log('Images created', result)
		},
		(error) => {
			if (typeof document !== 'undefined')
				document.write(`Error while creating Images: ${JSON.stringify(error)}`)
			console.error('Error while creating Images: ', error)
		}
	)
}

//Creates an Events Object in Parse DB
function sendEvent(swimDetected, numberSwimmers, drownDetected, serialNo) {
	let file = fs.readFileSync('../last_Image/last_Frame.jpg')
	const Events = Parse.Object.extend('Events')
	const event = new Events()

	event.set('swimDetected', swimDetected)
	event.set('numberSwimmers', numberSwimmers)
	event.set('drownDetected', drownDetected)
	event.set('serialNo', serialNo)
	event.set('image', new Parse.File('last_Frame.jpg', { base64: btoa(file) }))

	event.save().then(
		(result) => {
			if (typeof document !== 'undefined')
				document.write(`Images created: ${JSON.stringify(result)}`)
			console.log('Images created', result)
		},
		(error) => {
			if (typeof document !== 'undefined')
				document.write(`Error while creating Images: ${JSON.stringify(error)}`)
			console.error('Error while creating Images: ', error)
		}
	)
}

// constantly watches and creates Events and Image object when detected
// fs.watch('../last_Image/event.json', (event, filename) => {
// 	let prevEvent, currEvent
// 	if (filename) {
// 		let jsObj = require('../last_Image/event.json')
// 		// if (
// 		// 	jsObj.swimDetected === true ||
// 		// 	(jsObj.drownDetected === true && prevEvent !== jsObj)
// 		// ) {
// 			sendEvent(
// 				jsObj.swimDetected,
// 				parseInt(jsObj.numberSwimmers),
// 				jsObj.drownDetected,
// 				jsObj.serialNo
// 			)
// 			prevEvent = jsObj
// 		}
// 	}
// })

// Starts watching and creating Images objects when a LiveFeed object is created
const liveQuery = async () => {
	Parse.serverURL = 'https://optoswim.back4app.io' // Server URL
	Parse.initialize(
		'03Pq0kbLRvci8D3OV92OFIIbNidw3kZrGma2sruS', // Application ID
		'SMnIF1sMs1zdczYwWU1SikdLtfIu4IzcWYhBhEMf', // Javascript key
		'HSNUMKsUeTtNjTxOIPB2ct3FIiD6NMJp7yc5w9WW' // Master key (never use it in the frontend)
	)
	var client = await new Parse.LiveQueryClient({
		applicationId: '03Pq0kbLRvci8D3OV92OFIIbNidw3kZrGma2sruS',
		serverURL: 'wss://' + 'optoswim.back4app.io', // Example: 'wss://livequerytutorial.back4app.io'
		javascriptKey: 'SMnIF1sMs1zdczYwWU1SikdLtfIu4IzcWYhBhEMf',
		masterKey: 'HSNUMKsUeTtNjTxOIPB2ct3FIiD6NMJp7yc5w9WW',
	})
	await client.open()

	// Creates a new Query object to help us fetch MyCustomClass objects
	const query = new Parse.Query('LiveFeed')
	query.equalTo('liveFeed', true)

	var subscription = await client.subscribe(query)

	subscription.on('create', async () => {
		fs.watch('../last_Image/event.json', (event, filename) => {
			if (filename) {
				let jsObj = require('../last_Image/event.json')
				sendImage(
					jsObj.swimDetected,
					parseInt(jsObj.numberSwimmers),
					jsObj.drownDetected,
					jsObj.serialNo
				)
			}
		})
	})
}

liveQuery()
