//function to validate login credentials using stipulations given.

function validateForm()
	{
		var username = document.getElementById("name");
		var password = document.getElementById("password");
		
			if(username.value.trim()=="")
			{
				alert("Please fill in username.");
				return false;
			}
			else if(password.value.trim()=="")
			{
				alert("Please fill in password.");
				return false;
			}
			else if(password.value.trim().length<9)
			{
				alert("Password too short, must be 9 characters.");
				return false;
			}
			else if(password.value.trim().length>9)
			{
				alert("Password too long, must be 9 characters.");
				return false;
			}
			
			/*if (Number.isNaN(password)) 
			{
				alert("Invalid entry. Password must be numeric.");
				return false;
			}*/
			
			else
			{
				return true;
			}
		
	}