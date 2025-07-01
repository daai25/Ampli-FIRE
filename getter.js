require('dotenv').config();

const clientId = process.env.ID;
const clientSecret = process.env.SECRET;
/* curl -X POST "https://accounts.spotify.com/api/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \          
     -d "grant_type=client_credentials&client_id=ID&clientSecret=SECRET"
*/

async function getToken(){
      const request = await fetch("https://accounts.spotify.com/api/token", {
            method: "POST",
            headers: {
                  "Content-Type": "application/x-www-form-urlencoded",
            },
            body: `grant_type=client_credentials&client_id=${clientId}&client_secret=${clientSecret}`
      });
      const response = await request.json();
      return response['access_token'];
}


/* curl --request GET \
  --url https://api.spotify.com/v1/playlists/3cEYpjA9oz9GiPac4AsH4n \
  --header 'Authorization: Bearer 1POdFZRZbvb...qqillRxMr2z'
*/

async function getData(endpoint, token){
        const request = await fetch(endpoint, {
                headers: {
                        "Authorization" : `Bearer ${token}`,
                },
        });
        const response = await request.json();
        return response;
}

async function main(){
      const token = await getToken();
        console.log(token);
        const data = await getData("https://api.spotify.com/v1/playlists/5ezzEAFhsLp9ZYvCIrDJIu?si=2766faf4eb5d4abd", token);
        console.log(data.name);
        console.log("=========================");
        console.log(data.owner.id);
        console.log("=========================");
        console.log(data.description);
        console.log("=========================");
      for (const track of data.tracks.items) {
          process.stdout.write(`${track.track.href}: `);
          process.stdout.write(`${track.track.name} by `);
          track.track.artists.forEach(artist => {
            process.stdout.write(`${artist.name}, `);
          });
          console.log();
      }
}

main();
